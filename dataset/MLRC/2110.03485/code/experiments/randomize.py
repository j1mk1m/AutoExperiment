import os
import random
import pickle
import numpy as np

import torch
import torch.nn as nn
from pytorch_wavelets import DWTForward, DWTInverse

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class QuantitativeExperiment:
    def __init__(self, model, image, figure_type="a", mode="pixel", save_inter_masks=False):
        """
        args:
            model: pytorch model used with explanations
            image: image to be explained
            figure_type: type of analysis to be performed (a or b referring to Figure 7a/7b of the original CartoonX paper)
            mode: "pixel" or "wavelet" for experiments in pixel or wavelet domain
            save_inter_masks: if True, saves intermediate masks for each iteration
        """
        self.model = model
        self.image = image
        self.figure_type = figure_type
        self.masked_image = self.image.detach().clone()
        self.save_inter_masks = save_inter_masks
        self.distortion_scores = []
        self.resolution = 100

        if mode == "pixel":
            # Determine mean and standard deviation of image for random replacement
            self.std_pixel = torch.std(self.masked_image)
            self.mean_pixel = torch.mean(self.masked_image)
        elif mode == "wavelet":
            self.forward_dwt = DWTForward(
                J=5, mode="zero", wave="db3").to(device)
            self.inverse_dwt = DWTInverse(mode="zero", wave="db3").to(device)
            self.yl, self.yh = self.forward_dwt(self.image)
            # Determine mean and standard deviation of yl and yh for random replacement
            self.std_yl = torch.std(self.yl)
            self.mean_yl = torch.mean(self.yl)
            self.std_yh = []
            self.mean_yh = []
            for y in self.yh:
                self.std_yh.append(torch.std(y))
                self.mean_yh.append(torch.mean(y))

        else:
            raise ValueError(f"Mode {mode} not implemented.")

        if mode == "wavelet" and figure_type == "a":
            self.yl_orig = self.yl
            self.yh_orig = self.yh
            # Generate noise for yl coefficients with standard deviation std_yl
            self.yl = (torch.randn_like(self.yl) *
                       self.std_yl + self.mean_yl).to(device)
            # Generate noise for yh coefficients with standard deviation std_yh
            self.yh = []
            for count, y in enumerate(self.yh_orig):
                self.yh.append(
                    (torch.randn_like(y) * self.std_yh[count] + self.mean_yh[count]).to(device))
            self.masked_image = self.inverse_dwt((self.yl, self.yh))

        # Determine baseline prediction with original image
        with torch.no_grad():
            self.orig_prediction = nn.Softmax(dim=1)(model(image))
        self.orig_target = self.orig_prediction.max(1)[1].item()
        self.orig_probability = self.orig_prediction.max(1)[0].item()

    def execute(self):
        """
        Execute experiment
        """
        # determine distortion scores for masked images after every i-th percentile
        self.calculate_distortion()
        if self.save_inter_masks:
            self.save_intermediate_masks(0)
        for i in range(self.resolution):
            self.step()
            self.calculate_distortion()
            if self.save_inter_masks:
                self.save_intermediate_masks(i+1)
        return self.distortion_scores

    def save_intermediate_masks(self, i):
        """
        Save intermediate masks as numpy array
        args:
            i: current iteration number
        """
        save_path = f'logs/experiments/intermediate_masks/{self.figure_type}/{self.__class__.__name__}'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        intermediate_mask = self.masked_image.detach().clone()
        np.save(
            save_path + f'/intermediate_mask_{str(i).zfill(3)}.npy', intermediate_mask.cpu().numpy())

    def step(self):
        """
        Perform one step of the experiment (implemented in subclasses)
        """
        raise NotImplementedError("Step method not implemented.")

    def calculate_distortion(self, metric="l2"):
        """
        Calculate distortion of image based on masked image instance
        args:
            metric: metric to be used for distortion calculation
        """
        if metric == "label":
            with torch.no_grad():
                new_output = nn.Softmax(dim=1)(self.model(self.masked_image))
            new_probability = new_output[0, self.orig_target].item()
            distortion = (new_probability - self.orig_probability) ** 2
            self.distortion_scores.append(distortion)
        elif metric == "l2":
            with torch.no_grad():
                new_output = nn.Softmax(dim=1)(self.model(self.masked_image))
            assert len(
                self.orig_prediction.shape) == 2 and self.orig_prediction.shape[-1] == 1000, self.orig_prediction.shape
            distortion = torch.mean(torch.sqrt(
                ((self.orig_prediction - new_output)**2).sum(dim=-1))).item()
            self.distortion_scores.append(distortion)
        else:
            raise ValueError(f"Metric {metric} not implemented.")


class RandomPixel(QuantitativeExperiment):
    def __init__(self, **kwargs):
        """
        Baseline experiment for pixel domain
        """
        super().__init__(**kwargs)

        self.pixel_list = [(x, y) for x in range(self.masked_image.shape[2])
                           for y in range(self.masked_image.shape[3])]

        # Create fully randomized image
        if self.figure_type == "a":
            self.masked_image = torch.normal(
                self.mean_pixel, self.std_pixel, size=self.masked_image.shape).to(device)

    def step(self):
        """
        Perform one step of the experiment
        """
        for _ in range(np.ceil(self.image.shape[2] * self.image.shape[3] / self.resolution).astype(int)):
            if len(self.pixel_list) == 0:
                break
            # Randomly replace one pixel position with value sampled from normal distribution
            next_pixel_idx = np.random.randint(0, len(self.pixel_list))
            next_pixel = self.pixel_list.pop(next_pixel_idx)
            if self.figure_type == "a":
                self.masked_image[0, :, next_pixel[0], next_pixel[1]
                                  ] = self.image[0, :, next_pixel[0], next_pixel[1]]
            elif self.figure_type == "b":
                self.masked_image[0, :, next_pixel[0], next_pixel[1]] = torch.normal(
                    self.mean_pixel, self.std_pixel, size=(3,))


class RandomWavelet(QuantitativeExperiment):
    def __init__(self, **kwargs):
        """
        Baseline experiment for wavelet domain
        """
        super().__init__(**kwargs)

        # Setup list of wavelet coefficients to be replaced
        self.wavelet_list = [(self.yl, x, y) for x in range(
            self.yl.shape[2]) for y in range(self.yl.shape[3])]
        self.wavelet_list += [(yh_idx, idx, x, y) for yh_idx, yh in enumerate(self.yh)
                              for idx in range(yh.shape[2]) for x in range(yh.shape[3]) for y in range(yh.shape[4])]
        self.iternum = len(self.wavelet_list)

    def step(self):
        """
        Perform one step of the experiment
        """
        for _ in range(np.ceil(self.iternum / self.resolution).astype(int)):
            if len(self.wavelet_list) == 0:
                break
            next_wavelet_idx = np.random.randint(0, len(self.wavelet_list))
            next_wavelet = self.wavelet_list.pop(next_wavelet_idx)
            if self.figure_type == "a":
                # Restore original wavelet coefficient
                if len(next_wavelet) == 3:
                    self.yl[0, :, next_wavelet[1], next_wavelet[2]
                            ] = self.yl_orig[0, :, next_wavelet[1], next_wavelet[2]]
                elif len(next_wavelet) == 4:
                    self.yh[next_wavelet[0]][0, :, next_wavelet[1], next_wavelet[2], next_wavelet[3]
                                             ] = self.yh_orig[next_wavelet[0]][0, :, next_wavelet[1], next_wavelet[2], next_wavelet[3]]
            elif self.figure_type == "b":
                # Replace wavelet coefficient with random value
                if len(next_wavelet) == 3:
                    next_wavelet[0][0, :, next_wavelet[1], next_wavelet[2]] = torch.normal(
                        self.mean_yl, self.std_yl, size=(3,))
                elif len(next_wavelet) == 4:
                    self.yh[next_wavelet[0]][0, :, next_wavelet[1], next_wavelet[2], next_wavelet[3]] = torch.normal(
                        self.mean_yh[next_wavelet[0]], self.std_yh[next_wavelet[0]], size=(3,))
        self.masked_image = self.inverse_dwt((self.yl, self.yh))


class PixelRDE(QuantitativeExperiment):
    def __init__(self, mask_path, **kwargs):
        """
        Quantitative experiment for PixelRDE explanations
        """
        super().__init__(**kwargs)

        # Sort mask according to relevance score (per-pixel value)
        if mask_path.endswith(".npy"):
            self.mask = np.load(mask_path)
            if self.mask.shape[-1] == 3:
                self.mask = self.mask.mean(axis=-1)
        else:
            raise ValueError(f"Mask {mask_path} not supported.")

        self.indices = [(self.mask[x, y], x, y) for x in range(
            self.masked_image.shape[2]) for y in range(self.masked_image.shape[3])]
        random.shuffle(self.indices)
        self.indices.sort(key=lambda x: x[0], reverse=True)
        self.indices = [(i[1], i[2]) for i in self.indices]

        # Create fully randomized image
        if self.figure_type == "a":
            self.masked_image = torch.normal(
                self.mean_pixel, self.std_pixel, size=self.masked_image.shape).to(device)

    def step(self):
        """
        Perform one step of the experiment
        """
        for _ in range(np.ceil(self.image.shape[2] * self.image.shape[3] / self.resolution).astype(int)):
            if len(self.indices) == 0:
                break
            next_pixel = self.indices.pop(0)
            if self.figure_type == "a":
                # restore pixel value from original image
                self.masked_image[0, :, next_pixel[0], next_pixel[1]
                                  ] = self.image[0, :, next_pixel[0], next_pixel[1]]
            elif self.figure_type == "b":
                # replace pixel value with random value
                self.masked_image[0, :, next_pixel[0], next_pixel[1]] = torch.normal(
                    self.mean_pixel, self.std_pixel, size=(3,))


class CartoonX_exp(QuantitativeExperiment):
    def __init__(self, mask_path, **kwargs):
        """
        Quantitative experiment for CartoonX explanations
        """
        super().__init__(**kwargs)

        # Setup list of wavelet coefficients to be replaced
        with open(mask_path, "rb") as fp:  # Save DWTmask_cartoonX
            obj = pickle.load(fp)
            self.yl_mask, self.yh_mask = obj
        self.wavelet_list = [(self.yl_mask[0, x, y], self.yl, x, y) for x in range(
            self.yl.shape[2]) for y in range(self.yl.shape[3])]
        self.wavelet_list += [(self.yh_mask[yh_idx][0, idx, x, y], yh_idx, idx, x, y) for yh_idx, yh in enumerate(
            self.yh) for idx in range(yh.shape[2]) for x in range(yh.shape[3]) for y in range(yh.shape[4])]
        self.iternum = len(self.wavelet_list)
        random.shuffle(self.wavelet_list)
        # Sort wavelet coefficients according to first element of tuple (relevance score)
        self.wavelet_list.sort(key=lambda x: x[0], reverse=True)

    def step(self):
        """
        Perform one step of the experiment
        """
        for _ in range(np.ceil(self.iternum / 100).astype(int)):
            if len(self.wavelet_list) == 0:
                break
            next_wavelet = self.wavelet_list.pop(0)[1:]
            if self.figure_type == "a":
                # Restore original wavelet coefficient
                if len(next_wavelet) == 3:
                    self.yl[0, :, next_wavelet[1], next_wavelet[2]
                            ] = self.yl_orig[0, :, next_wavelet[1], next_wavelet[2]]
                elif len(next_wavelet) == 4:
                    self.yh[next_wavelet[0]][0, :, next_wavelet[1], next_wavelet[2], next_wavelet[3]
                                             ] = self.yh_orig[next_wavelet[0]][0, :, next_wavelet[1], next_wavelet[2], next_wavelet[3]]
            elif self.figure_type == "b":
                # Replace wavelet coefficient with random value
                if len(next_wavelet) == 3:
                    next_wavelet[0][0, :, next_wavelet[1], next_wavelet[2]] = torch.normal(
                        self.mean_yl, self.std_yl, size=(3,))
                elif len(next_wavelet) == 4:
                    self.yh[next_wavelet[0]][0, :, next_wavelet[1], next_wavelet[2], next_wavelet[3]] = torch.normal(
                        self.mean_yh[next_wavelet[0]], self.std_yh[next_wavelet[0]], size=(3,))
        self.masked_image = self.inverse_dwt((self.yl, self.yh)).clamp(0, 1)
