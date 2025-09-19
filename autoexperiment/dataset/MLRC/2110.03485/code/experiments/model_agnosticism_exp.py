import torch
import torchvision.models as models
import torch.nn as nn
from torchvision import transforms
import timm
from torch.utils.tensorboard import SummaryWriter

import numpy as np
from PIL import Image
import os
import sys
import argparse
import pickle
import cv2
import yaml
import matplotlib.pyplot as plt

import random
def set_seed(seed=42):
    # Set Python random seed
    random.seed(seed)
    
    # Set NumPy random seed
    np.random.seed(seed)
    
    # Set PyTorch random seed for CPU and GPU
    torch.manual_seed(seed)
    
    # Set CUDA random seed for all devices (if available)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
        
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(42) 


sys.path.insert(1, os.path.join(sys.path[0], '../'))
from cartoonx.cartoonX import CartoonX 

# Change direction to vit_explain
sys.path.insert(1, os.path.join(sys.path[1], 'vit-explain/'))
from vit_explain import show_mask_on_image
from vit_rollout import VITAttentionRollout


def get_attention_maps(imgdir, logdir, files):
    """
    args:
        imgdir: str, path to directory with images to explain
        storedir: str, path to directory where the results should be stored
        files: list, all the names of the images to explain
    """

    # Get device (use GPU if possible)
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    # Get classifier
    model = timm.create_model('deit_tiny_patch16_224',
                              pretrained=True).eval().to(device)

    for fname in files:
        print(f"Processing file: {fname} for attention maps")

        # Get image and transform to tensor
        path = os.path.join(imgdir, fname)
        x = Image.open(path)
        x = transforms.ToTensor()(x)

        # If image is grayscale, convert to RGB
        if x.shape[0] == 1:
            colored_image = np.zeros((3, x.shape[1], x.shape[2]))
            colored_image[0] = x
            colored_image[1] = x
            colored_image[2] = x
            x = colored_image
            x = torch.from_numpy(x).to(dtype=torch.float32)

        # Resize image to 224x224
        x = transforms.Resize(size=(224, 224))(x)
        x = x.to(device)

        # Get the attention maps
        rollout = VITAttentionRollout(model)
        mask = rollout(x.unsqueeze(0).detach())

        # Put mask on image for the attention rollout
        np_img = np.asarray(transforms.ToPILImage()(x.detach()))
        mask = cv2.resize(mask, (np_img.shape[1], np_img.shape[0]))
        masked_img = show_mask_on_image(np_img, mask)

        # Get both mask and image in RGB
        mask_rgb = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        masked_img_rgb = cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB)

        fname_split = fname.split(".")
        assert (
            fname_split[1] == "png"
            or fname_split[1] == "jpg"
            or fname_split[1] == "JPEG"
        ), "Image must be in .png or .jpg format."
        save_name = fname_split[0]

        log_path = os.path.join(logdir, save_name)

        # Plot attention rollout next to original image
        P = [
            (x, "Original image"),
            (masked_img_rgb, "Attention Rollout"),
        ]

        fig, axs = plt.subplots(1, 2, figsize=(8, 10))
        for idx, (img, title) in enumerate(P):
            args = {"cmap": "copper"} if idx > 0 else {}
            axs[idx].imshow(
                np.asarray(transforms.ToPILImage()(img)), **args
            )
            axs[idx].set_title(title, size=8)
            axs[idx].axis("off")
        
        # Save figure
        fig.savefig(
            os.path.join(log_path, "Attention_results.jpg"),
            bbox_inches="tight",
            transparent=True,
            pad_inches=0,
        )

        # Save the attention rollout explanation
        np.save(
            os.path.join(log_path, "attention_rollout.npy"),
            masked_img_rgb,
        )
        # Save the attention mask
        np.save(
            os.path.join(log_path, "attention_mask.npy"),
            mask_rgb,
        )


def cartoonX_res(imgdir, logdir, files, l1lambda=None):

    # Get device (use GPU if possible)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Get classifiers to explain: Vision Transformer and CNN
    vit_model = timm.create_model(
        'deit_tiny_patch16_224', pretrained=True).eval().to(device)
    cnn_model = models.mobilenet_v3_small(
        weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1).eval().to(device)

    with open(os.path.join(sys.path[0], "hparams_model_agnosticism_exp.yaml")) as f:
        HPARAMS = yaml.load(f, Loader=yaml.FullLoader)

    # Initialize wavelet RDE for CNN
    cartoonX_cnn = CartoonX(model=cnn_model, device=device, **HPARAMS)

    # Update hyperparameters if specified for the ViT
    if l1lambda is not None:
        HPARAMS["l1lambda"] = l1lambda

    # Initialize wavelet RDE for ViT
    cartoonX_vit = CartoonX(model=vit_model, device=device, **HPARAMS)

    # Create the logging directory if it doesn't exist
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    # Explain model decsision for each image in files
    for fname in files:
        print(f"Processing file: {fname} for cartoonX")
        # Get image and transform to tensor
        path = os.path.join(imgdir, fname)
        x = Image.open(path)
        x = transforms.ToTensor()(x)

        # If image is grayscale, convert to RGB
        if x.shape[0] == 1:
            colored_image = np.zeros((3, x.shape[1], x.shape[2]))
            colored_image[0] = x
            colored_image[1] = x
            colored_image[2] = x
            x = colored_image
            x = torch.from_numpy(x).to(dtype=torch.float32)

        # Resize image to 224x224
        x = transforms.Resize(size=(224, 224))(x)
        x = x.to(device)

        # Get prediction for x from vit
        output_vit = vit_model(x.unsqueeze(0).detach())
        max_idx_vit = nn.Softmax(dim=1)(output_vit).max(1)[1].item()

        # Get prediction for x from cnn
        output_cnn = cnn_model(x.unsqueeze(0).detach())
        max_idx_cnn = nn.Softmax(dim=1)(output_cnn).max(1)[1].item()

        # Get Cartoon Explanation for x
        exp_cartoonX_vit, mask_vit, logs_vit, _ = cartoonX_vit(
            x.unsqueeze(0), target=max_idx_vit, path=path)
        exp_cartoonX_cnn, mask_cnn, logs_cnn, _ = cartoonX_cnn(
            x.unsqueeze(0), target=max_idx_cnn, path=path)

        fname_split = fname.split(".")
        assert (
            fname_split[1] == "png"
            or fname_split[1] == "jpg"
            or fname_split[1] == "JPEG"
        ), "Image must be in .png or .jpg format."
        save_name = fname_split[0]

        if not os.path.exists(os.path.join(logdir, save_name)):
            os.makedirs(os.path.join(logdir, save_name))

        log_path = os.path.join(logdir, save_name)

        # Plot explanations next to original image
        P = [
            (x, "Original image"),
            (exp_cartoonX_cnn, "CartoonX for CNN"),
            (exp_cartoonX_vit, "CartoonX for ViT"),
        ]
        fig, axs = plt.subplots(1, 3, figsize=(10, 10))
        for idx, (img, title) in enumerate(P):
            args = {"cmap": "copper"} if idx > 0 else {}
            axs[idx].imshow(
                np.asarray(transforms.ToPILImage()(img)), **args
            )
            axs[idx].set_title(title, size=8)
            axs[idx].axis("off")
        
        # Save figure
        fig.savefig(
            os.path.join(log_path, "CartoonX_results.jpg"),
            bbox_inches="tight",
            transparent=True,
            pad_inches=0,
        )

        # Log to tensorboard and save data to storedir
        print(f"Final ViT loss sparsity: {logs_vit['loss-sparsity'][-1]}")
        writer = SummaryWriter(log_path)
        for i in range(len(logs_vit["loss"])):
            writer.add_scalar("Loss ViT", logs_vit["loss"][i], global_step=i)
            writer.add_scalar(
                "Loss ViT (Sparsity Component)",
                logs_vit["loss-sparsity"][i],
                global_step=i,
            )
            writer.add_scalar(
                "Loss ViT (Distortion Component)",
                logs_vit["loss-distortion"][i],
                global_step=i,
            )
            writer.add_scalar(
                "L1-Norm ViT", logs_vit["l1-norm"][i], global_step=i
            )
            writer.add_scalar(
                "Distortion ViT", logs_vit["distortion"][i], global_step=i
            )

        for i in range(len(logs_cnn["loss"])):
            writer.add_scalar("Loss CNN", logs_cnn["loss"][i], global_step=i)
            writer.add_scalar(
                "Loss CNN (Sparsity Component)",
                logs_vit["loss-sparsity"][i],
                global_step=i,
            )
            writer.add_scalar(
                "Loss CNN (Distortion Component)",
                logs_cnn["loss-distortion"][i],
                global_step=i,
            )
            writer.add_scalar(
                "L1-Norm CNN", logs_cnn["l1-norm"][i], global_step=i
            )
            writer.add_scalar(
                "Distortion CNN", logs_cnn["distortion"][i], global_step=i
            )
        writer.flush()
        writer.close()

        # Save the original image
        np.save(
            os.path.join(log_path, "original_image.npy"),
            np.asarray(transforms.ToPILImage()(x.cpu())),
        )

        # Save the vit cartoonX explanation
        np.save(
            os.path.join(log_path, "exp_cartoonX_vit.npy"),
            np.asarray(transforms.ToPILImage()(exp_cartoonX_vit.cpu())),
        )
        # Save the cnn cartoonX explanation
        np.save(
            os.path.join(log_path, "exp_cartoonX_cnn.npy"),
            np.asarray(transforms.ToPILImage()(exp_cartoonX_cnn.cpu())),
        )

        # Save DWT Mask from the ViT
        with open(os.path.join(log_path, "DWTmask_vit.pickle"), "wb") as fp:
            pickle.dump(mask_vit, fp)

        # Save DWT Mask from the CNN
        with open(os.path.join(log_path, "DWTmask_cnn.pickle"), "wb") as fp:
            pickle.dump(mask_cnn, fp)

        # Save the prediction of both clasifiers
        with open(os.path.join(log_path, "predictions.txt"), "w") as f:
            f.write(f"{max_idx_vit} {max_idx_cnn}")


def main(n_images, imgdir, logdir, l1lambda):
    """
    args:
        n_images: int, maximum number of images from imgdir to explain
        imgdir: str, path to directory with images to explain
        storedir: str, path to directory where the results should be stored
        l1lambda: float, lambda k value for CartoonX for the Vision Transformer (ViT)
    """

    # Get files of images
    images = [f for f in os.listdir(imgdir) if f.lower().endswith(
        ".jpg") or f.lower().endswith(".png") or f.lower().endswith(".jpeg")]
    images = np.random.permutation(images)  # shuffle files
    images = images[:n_images] if len(
        images) > n_images else images  # only use n_images

    # Run CartoonX
    cartoonX_res(imgdir=imgdir, logdir=logdir, files=images, l1lambda=l1lambda)
    # Run attention rollout
    get_attention_maps(imgdir=imgdir, logdir=logdir, files=images)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--imgdir", type=str, help="Directory of images to explain", default="images/imagenet_sample")
    parser.add_argument(
        "--logdir", type=str, help="Directory where explanations are logged", default="logs/experiment2")
    parser.add_argument("--n_images", "-n", type=int,
                        help="Number of images to explain", default=1)
    parser.add_argument("--lambda_vit", type=float,
                        help="l1lambda ViT", default=10)

    args = parser.parse_args()
    main(n_images=args.n_images, imgdir=args.imgdir,
         logdir=args.logdir, l1lambda=args.lambda_vit)
