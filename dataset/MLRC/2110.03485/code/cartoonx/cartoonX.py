import sys
import cv2
import numpy as np
import torch
from torchvision import transforms
from pytorch_wavelets import DWTForward, DWTInverse


class CartoonX:
    def __init__(self, model, device, batch_size, num_steps, step_size, l1lambda, wave, mode, J,
                 preoptimization_step_size=0.01, distortion_measure="label", obfuscation_strategy="gaussian-adaptive-noise", init_mask="ones"):
        """
        args:
            model: classifier to be explained
            device: gpu or cpu
            batch_size: int - number of samples to approximate expected distortion
            num_steps: int - number of optimization steps for mask
            step_size: float - step size for adam optimizer on mask
            l1lambda: float - Lagrange multiplier for l1 norm of mask ("lambda k" in paper)
            wave: str - wave type for DWT e.g. "db3"
            mode: str - mode for DWT e.g. "zero"
            J: int - number of scales for DWT
            preoptimization_step_size: float - step size used for preoptimization of mask
            distortion_measure: str - identifier of distortion measure function; either "label", "l2", "kl-divergence", or "weighted-l2"
            obfuscation_strategy: str - either "gaussian-adaptive-noise" or "zero"
            init_mask: str - "ones" or "rand"
        """
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.step_size = step_size
        self.l1lambda = l1lambda
        self.forward_dwt = DWTForward(J=J, mode=mode, wave=wave).to(device)
        self.inverse_dwt = DWTInverse(mode=mode, wave=wave).to(device)
        self.softmax = torch.nn.Softmax(dim=1)
        self.preoptimization_step_size = preoptimization_step_size
        self.distortion_measure = distortion_measure
        self.obfuscation_strategy = obfuscation_strategy
        self.init_mask = init_mask

    def step(self, std_yl, mean_yl, std_yh, mean_yh, yl, yh, s_yl, s_yh, score, target, num_mask_entries):
        # Generate noise for yl coefficients with standard deviation std_yl
        n_yl = std_yl * torch.randn((self.batch_size, *yl.shape[1:]),
                                    dtype=torch.float32,
                                    device=self.device,
                                    requires_grad=False) + mean_yl

        # Get obfuscation for yl coefficients
        obf_yl = s_yl*yl + (1-s_yl)*n_yl

        # Initialize list of obfuscations for yh coefficients
        obf_yh = []
        for count, y in enumerate(yh):

            # Generate noise for yh coefficients with standard deviation in  std_yh and mean in_mean_yh
            n_y = std_yh[count] * torch.randn((self.batch_size, *y.shape[1:]),
                                              dtype=torch.float32,
                                              device=self.device,
                                              requires_grad=False) + mean_yh[count]

            # Get obfuscation for yh coefficients
            obf_yh.append(s_yh[count]*y + (1-s_yh[count])*n_y)

        # Inverse wavelet transform of obfuscation
        assert obf_yl.shape == (
            self.batch_size, *yl.shape[1:]) and obf_yh[0].shape == (self.batch_size, *yh[0].shape[1:])
        z = self.inverse_dwt((obf_yl, obf_yh))
        # We clamp overflowing values after inverse DWT back to [0,1]
        z = z.clamp(0, 1)

        # Get model score for the obfuscation z
        new_preds = self.model(z)
        assert len(new_preds.shape) == 2

        distortion = self.get_distortion(score, new_preds, target)

        # Compute sparsity loss for s_yl
        sparsity = torch.sum(torch.abs(s_yl))

        # Compute sparsity loss for masks in s_yh
        for s in s_yh:
            sparsity += torch.sum(torch.abs(s))

        # Divide sparsity loss with number of mask entries: adjust lambda if you don't want to normalize
        sparsity /= num_mask_entries

        is_same_classification = torch.argmax(
            torch.mean(new_preds, dim=0)).item() == target

        return distortion, sparsity, is_same_classification

    def get_distortion(self, score, new_preds, target):
        # Compute distortion between score for obfuscation z and and score for original image
        if self.distortion_measure == "label" or self.distortion_measure == "maximize-target":
            # Compute distortion in the predicted target label (for maximize-target score=1 otherwise "score=labelprobabilty")
            new_scores = self.softmax(new_preds)[:, target]
            # Approximate expected distortion with simple Monte-Carlo estimate
            distortion = torch.mean((score - new_scores)**2)
        elif self.distortion_measure == "l2":
            # Computes distortion as squared ell_2 norm between model outputs
            new_scores = self.softmax(new_preds)
            # Approximate expected distortion with simple Monte-Carlo estimate
            assert len(score.shape) == 2 and score.shape[-1] == 1000, score
            distortion = torch.mean(torch.sqrt(
                ((score - new_scores)**2).sum(dim=-1)))
        elif self.distortion_measure == "kl-divergence":
            new_scores = self.softmax(new_preds)
            # Compute average kl-divergence for prediction by obfuscations to original prediction
            distortion = torch.mean(
                (new_scores * torch.log(new_scores/score)).sum(dim=-1))
        elif self.distortion_measure == "weighted-l2":
            new_scores = self.softmax(new_preds)
            distortion = self.C * \
                torch.mean(torch.sqrt(((score - new_scores)**2).sum(dim=-1)))
        else:
            raise NotImplementedError(
                f"distortion measure {self.distortion_measure} was not implemented.")
        return distortion

    def get_scaled_mask(self, mask, size, epsilon):
        if isinstance(size, int):
            size = (size, size)
        s = transforms.ToTensor()(mask).to(self.device)
        s = transforms.Resize(size=size)(s)
        s = (s > 0).to(torch.float32)
        s = torch.where(s == 0, epsilon, s)
        return s

    def initialize_dwt_mask(self, yl, yh, path, epsilon=1):
        if self.init_mask == "ones":
            # Get mask for yl coefficients
            yl.requires_grad_(False).to(self.device)
            s_yl = torch.ones((1, *yl.shape[2:]),
                              dtype=torch.float32,
                              device=self.device,
                              requires_grad=False) * epsilon
            s_yl.requires_grad_(True)

            # Get mask for yh coefficients
            s_yh = []
            for y in yh:
                y.requires_grad_(False).to(self.device)
                s = torch.ones((1, *y.shape[2:]),
                               dtype=torch.float32,
                               device=self.device,
                               requires_grad=False) * epsilon
                s.requires_grad_(True)
                s_yh.append(s)

        elif self.init_mask == "zeros":
            # Get mask for yl coefficients
            yl.requires_grad_(False).to(self.device)
            s_yl = torch.zeros((1, *yl.shape[2:]),
                               dtype=torch.float32,
                               device=self.device,
                               requires_grad=True)

            # Get mask for yh coefficients
            s_yh = []
            for y in yh:
                y.requires_grad_(False).to(self.device)
            s_yh.append(torch.zeros((1, *y.shape[2:]),
                                    dtype=torch.float32,
                                    device=self.device,
                                    requires_grad=True))

        elif self.init_mask == "rand":
            # Get mask for yl coefficients
            yl.requires_grad_(False).to(self.device)
            s_yl = torch.rand((1, *yl.shape[2:]),
                              dtype=torch.float32,
                              device=self.device,
                              requires_grad=True)

            # Get mask for yh coefficients
            s_yh = []
            for y in yh:
                y.requires_grad_(False).to(self.device)
                s_yh.append(torch.rand((1, *y.shape[2:]),
                                       dtype=torch.float32,
                                       device=self.device,
                                       requires_grad=True))

        elif self.init_mask == "foreground":
            # Forground/background segmentation using implmenentation from https://www.geeksforgeeks.org/python-foreground-extraction-in-an-image-using-grabcut-algorithm/
            orig_image = cv2.imread(path)
            mask = np.zeros(orig_image.shape[:2], np.uint8)
            backgroundModel = np.zeros((1, 65), np.float64)
            foregroundModel = np.zeros((1, 65), np.float64)
            rectangle = (1, 1, *orig_image.shape[:2])

            # Generate the foreground mask
            cv2.grabCut(orig_image, mask, rectangle,
                        backgroundModel, foregroundModel,
                        3, cv2.GC_INIT_WITH_RECT)
            mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

            # Scale the foreground mask to fit the wavelet coefficients.
            s_yl = self.get_scaled_mask(mask2, yl.shape[-2:], epsilon)
            s_yl.requires_grad_(True)
            s_yh = []
            for y in yh:
                s = self.get_scaled_mask(mask2, y.shape[-2:], epsilon)
                s = s.repeat(3, 1, 1)[None, :]
                s.requires_grad_(True)
                s_yh.append(s)

        else:
            raise NotImplementedError(
                f"mask initialization {self.init_mask} was not implemented.")

        # Get total number of mask entries
        num_mask_entries = s_yl.shape[-1] * s_yl.shape[-2]
        for s in s_yh:
            num_mask_entries += s.shape[-1] * s.shape[-2] * s.shape[-3]

        return s_yl, s_yh, num_mask_entries

    def __call__(self, x, target, path, save_mask_after=[], preoptimize=False):
        # Assert image has shape (1,C,H,W)
        assert len(x.shape) == 4
        x.requires_grad_(False)

        # Initialize list for logs
        logs = {"l1-norm": [], "distortion": [], "loss": [],
                "loss-sparsity": [], "loss-distortion": []}

        """
        Do forward DWT
        """

        # Do forward DWT for images
        # yl.shape = (1, 3, ?, ?) yh[i].shape = (1, 3, 3 ?, ?)
        yl, yh = self.forward_dwt(x)

        # Get DWT for greyscale image
        yl_grey, yh_grey = self.forward_dwt(x.sum(dim=1, keepdim=True)/3)

        """
        Initialize obfuscation strategy
        """
        if self.obfuscation_strategy == "gaussian-adaptive-noise":
            # Compute standard deviation and mean for adaptive Gaussian noise (this is the obfuscation strategy we use in our paper)
            std_yl = torch.std(yl)
            mean_yl = torch.mean(yl)
            std_yh = []
            mean_yh = []
            for y in yh:
                std_yh.append(torch.std(y))
                mean_yh.append(torch.mean(y))
        elif self.obfuscation_strategy == "zero":
            std_yl = 0
            mean_yl = 0
            std_yh = []
            mean_yh = []
            for y in yh:
                std_yh.append(0)
                mean_yh.append(0)
        else:
            raise NotImplementedError(
                f"Obfuscation strategy {self.obfuscation_strategy} was not implemented")

        """
        Get score for original image
        """
        if self.distortion_measure == "label":
            # Measure distortion as squared difference in target label from obfuscation to original
            score = self.softmax(self.model(x.detach()).detach())[
                :, target].detach()
        elif self.distortion_measure == "maximize-target":
            # Measure distortion as squared difference in target label from obfuscation to 1, i.e. the maximal possible score
            score = 1
        elif self.distortion_measure == "l2":
            # Measure distortion as
            assert target is None
            score = self.softmax(self.model(x.detach()).detach())
        elif self.distortion_measure == "kl-divergence":
            score = self.softmax(self.model(x.detach()).detach())
            assert target is None
        elif self.distortion_measure == "weighted-l2":
            score = target[0]
            self.C = target[1]
        else:
            raise NotImplementedError(
                f"distortion measure {self.distortion_measure} was not implemented.")

        """
        Initialize mask and pre-optimize.
        """
        step_size = self.preoptimization_step_size
        if preoptimize:
            assert self.init_mask in [
                "foreground", "ones"], "preoptimization only works with foreground or ones mask initialization"
            for epsilon in np.arange(1, 0, -step_size):
                s_yl, s_yh, num_mask_entries = self.initialize_dwt_mask(
                    yl, yh, path, epsilon)
                distortion, sparsity, is_same_classification = self.step(
                    std_yl, mean_yl, std_yh, mean_yh, yl, yh, s_yl, s_yh, score, target, num_mask_entries)
                if not is_same_classification:
                    s_yl, s_yh, num_mask_entries = self.initialize_dwt_mask(
                        yl, yh, path, epsilon + step_size)
                    distortion, sparsity, is_same_classification = self.step(
                        std_yl, mean_yl, std_yh, mean_yh, yl, yh, s_yl, s_yh, score, target, num_mask_entries)
                    print(
                        f"FINAL preoptimization with epsilon = {epsilon + step_size} and distortion = {distortion}, same classification = {is_same_classification}")
                    break
        else:
            if self.init_mask == "foreground":
                s_yl, s_yh, num_mask_entries = self.initialize_dwt_mask(
                    yl, yh, path, epsilon=0)
            else:
                s_yl, s_yh, num_mask_entries = self.initialize_dwt_mask(
                    yl, yh, path, epsilon=1)

        """
        Initialize Optimizer for mask
        """
        optimizer = torch.optim.Adam([s_yl]+s_yh, lr=self.step_size)

        """
        Start optimizing masks
        """
        intermediate_masks = []
        for i in range(self.num_steps):

            """
            Compute distortion and sparasity
            """
            distortion, sparsity, _ = self.step(std_yl, mean_yl, std_yh, mean_yh, yl,
                                                yh, s_yl, s_yh, score, target, num_mask_entries)

            # Compute loss
            loss = distortion + self.l1lambda * sparsity

            # Log loss terms
            logs["distortion"].append(distortion.detach().item())
            logs["l1-norm"].append(sparsity.detach().item())
            logs["loss"].append(loss.detach().item())
            logs["loss-sparsity"].append(
                (self.l1lambda * sparsity).detach().item())
            logs["loss-distortion"].append(distortion.detach().item())

            # Perform optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Clamp masks into [0,1]
            with torch.no_grad():
                s_yl.clamp_(0, 1)
                for s_y in s_yh:
                    s_y.clamp_(0, 1)

            # Save intermediate masks
            if i in save_mask_after:
                mask = [s_yl.cpu().detach().numpy(), [
                    s.cpu().detach().numpy() for s in s_yh]]
                intermediate_masks.append(mask)

        """
        Invert DWT mask back to pixel space
        """
        cartoonX = self.inverse_dwt((s_yl.detach()*yl_grey,
                                     [s.detach()*y for s, y in zip(s_yh, yh_grey)]))
        # We take absolute value since 0 values in pixel space needs to be smallest values.
        # We also clamp into 0,1 in case there was an overflow, i.e. pixel values larger than 1 after the inverse dwt
        cartoonX = cartoonX.squeeze(0).clamp_(0, 1)
        mask = [s_yl.cpu().detach().numpy(), [s.cpu().detach().numpy()
                                              for s in s_yh]]

        # CartoonX is the mask in pixel/image space, mask is the mask in DWT space
        return cartoonX, mask, logs, intermediate_masks
