import os
import sys
import yaml
import argparse
from datetime import datetime
import pickle
import timm

import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter

import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

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

sys.path.insert(1, os.path.join(sys.path[0], "../"))

from cartoonX import CartoonX
from pixelRDE import PixelRDE

from utils import hparams_to_str

# Get current time for logging
now = datetime.now()
current_time = now.strftime("%d/%m/%Y %H:%M:%S")

# Get list of imagenet labels to convert prediction to string label
LABEL_LIST = tuple(
    open(os.path.join(sys.path[0], "imagenet_labels.txt")).read().split("\n")
)
LABEL_LIST = [
    x.replace("{", "")
    .replace("'", "")
    .replace(",", "")
    .replace("-", " ")
    .replace("_", " ")
    for x in LABEL_LIST
]


def main(
    model,
    hparams,
    imgdir,
    logdir,
    resize_images,
    n_images,
    shuffle_images,
    lambda_cartoonx,
    lambda_pixelrde,
    save_mask_after,
    preoptimize,
    save_files,
):
    """
    Main function to explain model decisions for given image(s) using CartoonX and/or PixelRDE.
    args:
        model: str, name of model to explain (one of "VGG16", "mobile-net", "vit", "deit")
        hparams: str, path to yaml file with hyperparameters
        imgdir: str, path to directory with images to explain
        logdir: str, path to directory to save tensorboard logs (and additional results)
        resize_images: int, size to resize images to
        n_images: int, maximum number of images from imgdir to explain
        shuffle_images: bool, whether to shuffle images before explaining
        lambda_cartoonx: float, lambda k value for CartoonX
        lambda_pixelrde: float, lambda k value for PixelRDE
        save_mask_after: str, comma-separated string of iterations to save mask after
        preoptimize: bool, whether apply preoptimize heuristic to masks
        save_files: bool, whether to save additional files (masks, images, etc.)
    """
    # Get device (use GPU if possible)
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    # Get classifier to explain
    if model == "VGG16":
        model = (
            models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).eval().to(device)
        )
    elif model == "mobile-net":
        model = (
            models.mobilenet_v3_small(
                weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
            )
            .eval()
            .to(device)
        )
    elif model == "deit":
        model = (
            timm.create_model("deit_tiny_patch16_224", pretrained=True)
            .eval()
            .to(device)
        )
    else:
        raise ValueError(f"Model {model} not implemented.")

    # Get hyperparameters for wavelet RDE and pixel RDE
    with open(os.path.join(sys.path[0], hparams)) as f:
        HPARAMS_CARTOONX = yaml.load(f, Loader=yaml.FullLoader)["CartoonX"]
    with open(os.path.join(sys.path[0], hparams)) as f:
        HPARAMS_PIXEL_RDE = yaml.load(f, Loader=yaml.FullLoader)["PixelRDE"]

    # Update hyperparameters if specified
    if lambda_cartoonx is not None:
        HPARAMS_CARTOONX["l1lambda"] = int(lambda_cartoonx)
    if lambda_pixelrde is not None:
        HPARAMS_PIXEL_RDE["l1lambda"] = int(lambda_pixelrde)

    # Initialize wavelet RDE and pixel RDE
    cartoonX = CartoonX(model=model, device=device, **HPARAMS_CARTOONX)
    pixelRDE = PixelRDE(model=model, device=device, **HPARAMS_PIXEL_RDE)

    # Get files of images
    files = [
        f
        for f in os.listdir(imgdir)
        if f.lower().endswith(".jpg")
        or f.lower().endswith(".png")
        or f.lower().endswith(".jpeg")
    ]
    if shuffle_images:
        files = np.random.permutation(files)  # shuffle files
    files = (
        files[:n_images] if len(files) > n_images else files
    )  # only use n_images

    # Convert string of ints to list of ints
    if save_mask_after != "":
        save_mask_after = [int(s) for s in save_mask_after.split(',')]
    else:
        save_mask_after = []

    # Explain model decsision for each image in files
    for fname in files:
        print(f"Processing file: {fname}")
        # Get image and transform to tensor
        path = os.path.join(imgdir, fname)
        x = Image.open(path)
        x = transforms.ToTensor()(x)

        if x.shape[0] == 1: # if image is grayscale, convert to RGB
            colored_image = np.zeros((3, x.shape[1], x.shape[2]))
            colored_image[0] = x
            colored_image[1] = x
            colored_image[2] = x
            x = colored_image
            x = torch.from_numpy(x).to(dtype=torch.float32)
        
        x = transforms.Resize(size=(resize_images, resize_images))(x)
        x = x.to(device)

        # Get prediction for x
        output = model(x.unsqueeze(0).detach())
        max_idx = nn.Softmax(dim=1)(output).max(1)[1].item()
        label = LABEL_LIST[max_idx]


        # Get explanation for x
        exp_cartoonX, DWTmask_cartoonX, logs_cartoonX, intermediate_mask_cartoonx = cartoonX(
            x.unsqueeze(0),
            target=max_idx,
            path=path,
            save_mask_after=save_mask_after,
            preoptimize=preoptimize,
        )
        exp_pixelRDE, logs_pixelRDE, intermediate_mask_rde = pixelRDE(
            x.unsqueeze(0),
            target=max_idx, save_mask_after=save_mask_after
        )

        # Plot explanations next to original image
        P = [
            (x, f"Pred:{label}"),
            (exp_cartoonX, "CartoonX"),
            (exp_pixelRDE, "Pixel RDE"),
        ]
        fig, axs = plt.subplots(1, 3, figsize=(10, 10))
        for idx, (img, title) in enumerate(P):
            args = {"cmap": "copper"} if idx > 0 else {}
            axs[idx].imshow(
                np.asarray(transforms.ToPILImage()(img)), vmin=0, vmax=255, **args
            )
            axs[idx].set_title(title, size=8)
            axs[idx].axis("off")

        # Strip file extension from fname to create folder names
        fname_split = fname.split(".")
        assert (
            fname_split[1] == "png"
            or fname_split[1] == "jpg"
            or fname_split[1] == "JPEG"
        ), "Image must be in .png or .jpg format."
        save_name = fname_split[0]

        # Things to save in the folder name
        hparams_str = {
            "CNX_l1lambda": HPARAMS_CARTOONX["l1lambda"],
            "RDE_l1lambda": HPARAMS_PIXEL_RDE["l1lambda"],
            "mask": HPARAMS_CARTOONX["init_mask"][0],
            "preopt": preoptimize
        }

        save_name += hparams_to_str(hparams_str)
        log_path = os.path.join(logdir, save_name)

        # Log to tensorboard and save data to logdir
        print(f"Final distortion loss: {logs_cartoonX['loss-distortion'][-1]}")
        writer = SummaryWriter(log_path)
        writer.add_figure(f"Explanations", fig)
        for i in range(len(logs_cartoonX["loss"])):
            writer.add_scalar("Loss CartoonX", logs_cartoonX["loss"][i], global_step=i)
            writer.add_scalar(
                "Loss CartoonX (Sparsity Component)",
                logs_cartoonX["loss-sparsity"][i],
                global_step=i,
            )
            writer.add_scalar(
                "Loss CartoonX (Distortion Component)",
                logs_cartoonX["loss-distortion"][i],
                global_step=i,
            )
            writer.add_scalar(
                "L1-Norm CartoonX", logs_cartoonX["l1-norm"][i], global_step=i
            )
            writer.add_scalar(
                "Distortion CartoonX", logs_cartoonX["distortion"][i], global_step=i
            )
        for i in range(len(logs_pixelRDE["loss"])):
            writer.add_scalar("Loss PixelRDE", logs_pixelRDE["loss"][i], global_step=i)
            writer.add_scalar(
                "L1-Norm PixelRDE", logs_pixelRDE["l1-norm"][i], global_step=i
            )
            writer.add_scalar(
                "Distortion PixelRDE", logs_pixelRDE["distortion"][i], global_step=i
            )
        writer.flush()
        writer.close()

        fig.savefig(
            os.path.join(log_path, "figure.jpg"),
            bbox_inches="tight",
            transparent=True,
            pad_inches=0,
        )

        if save_files:
            # Save additional results to log_path
            np.save(os.path.join(log_path, "original_image.npy"), x.cpu().detach().numpy())
            np.save(
                os.path.join(log_path, "exp_cartoonX.npy"),
                exp_cartoonX.cpu().detach().numpy(),
            )
            np.save(
                os.path.join(log_path, "exp_pixelRDE.npy"),
                exp_pixelRDE.cpu().detach().numpy(),
            )
            with open(
                os.path.join(log_path, "DWTmask_cartoonX.pickle"), "wb"
            ) as fp:  # Save DWTmask_cartoonX
                pickle.dump(DWTmask_cartoonX, fp)
            with open(os.path.join(log_path, "pred.txt"), "w") as f:  # Save prediction
                f.write(f"{label}, {max_idx}")
            with open(
                os.path.join(log_path, "intermediate_mask_cartoonx.pickle"), "wb"
            ) as fp:  # Save intermediate masks
                pickle.dump(intermediate_mask_cartoonx, fp)
            with open(
                os.path.join(log_path, "intermediate_mask_rde.pickle"), "wb"
            ) as fp:  # Save intermediate masks
                pickle.dump(intermediate_mask_rde, fp)

            # Log the hparams file to check later what hparams were used
            with open(os.path.join(log_path, "hparams.yaml"), "w") as f:
                yaml.dump(
                    {"CartoonX": HPARAMS_CARTOONX, "PixelRDE": HPARAMS_PIXEL_RDE},
                    f,
                    default_flow_style=False,
                )


if __name__ == "__main__":
    # Parse CLI arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, help="Model to be explained.", default="mobile-net"
    )
    parser.add_argument(
        "--hparams",
        type=str,
        help="Hyperparameter configuration",
        default="hparams.yaml",
    )
    parser.add_argument(
        "--imgdir",
        type=str,
        help="Directory of images to explain.",
        default="images/imagenet_sample",
    )
    parser.add_argument(
        "--logdir",
        type=str,
        help="Directory where explanations are logged",
        default="logs/explanations/main",
    )
    parser.add_argument(
        "--resize_images", type=int, help="Resolution of the image", default=256
    )
    parser.add_argument(
        "--n_images", "-n", type=int, help="Number of images to explain", default=1
    )
    parser.add_argument("--shuffle_images", dest="shuffle_images", action="store_true")
    parser.add_argument("--lambda_cartoonx", help="l1lambda CartooNX", default=None)
    parser.add_argument("--lambda_pixelrde", help="l1lambda Pixel RDE", default=None)
    parser.add_argument(
        "--save_mask_after", type=str, help="saves the mask after x steps, list of ints", default=""
    )
    parser.add_argument(
        "--preoptimize",
        dest="preoptimize",
        help="preoptimization for better mask initialization for faster convergence",
        action="store_true",
    )
    # add argument to save files or not
    parser.add_argument(
        "--save_files",
        dest="save_files",
        help="save additional result files",
        default=True,
    )
    args = parser.parse_args()

    main(
        model=args.model,
        hparams=args.hparams,
        imgdir=args.imgdir,
        logdir=args.logdir,
        resize_images=args.resize_images,
        n_images=args.n_images,
        shuffle_images=args.shuffle_images,
        lambda_cartoonx=args.lambda_cartoonx,
        lambda_pixelrde=args.lambda_pixelrde,
        save_mask_after=args.save_mask_after,
        preoptimize=args.preoptimize,
        save_files=args.save_files,
    )
