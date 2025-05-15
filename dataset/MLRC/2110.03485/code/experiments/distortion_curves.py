import os, sys
import argparse
import pickle
import numpy as np
from tqdm import tqdm

import timm
import torch
import torchvision.models as models

from randomize import RandomPixel, RandomWavelet, PixelRDE, CartoonX_exp

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

sys.path.insert(1, os.path.join(sys.path[0], "../"))


def distortion_curves(
    figure_type,
    logdir,
    image_paths,
    storedir,
    l1lambda,
    save_inter_masks,
    experiment_type="reproduction",
):
    """
    args:
        figure_type: str, type of figure to be plotted ("a" or "b")
        logdir: str, path to directory containing the images
        image_paths: list, list of image file names
        storedir: str, path to directory where the results should be stored
        l1lambda: float, lambda value used for CartoonX and Pixel RDE experiments (e.g. "CNX20_RDE4")
        save_inter_masks: bool, whether to save intermediate masks (only one image will be considered)
        experiment_type: str, type of experiment to be performed ("reproduction" or "ViT")
    """
    global device

    # Set up experiment and load model
    if experiment_type == "reproduction":
        model = models.mobilenet_v3_small(pretrained=True).eval().to(device)
        random_pixel_scores_total = []
        random_wavelet_scores_total = []
        pixel_rde_scores_total = []
        cartoonX_scores_total = []
    elif experiment_type == "ViT":
        model_cnn = models.mobilenet_v3_small(pretrained=True).eval().to(device)
        model_vit = (
            timm.create_model("deit_tiny_patch16_224", pretrained=True)
            .eval()
            .to(device)
        )
        random_pixel_scores_CNN_total = []
        random_wavelet_scores_CNN_total = []
        random_pixel_scores_ViT_total = []
        random_wavelet_scores_ViT_total = []
        cartoonX_scores_CNN_total = []
        cartoonX_scores_ViT_total = []
        attention_masks_scores_total = []
    else:
        raise ValueError("Experiment type not recognized!")

    print(
        f"Calcululating distortion curve for {len(image_paths)} images with method {figure_type}!"
    )

    if save_inter_masks:
        print("Saving intermediate masks!")
        image_paths = image_paths[:1]

    for image_path in tqdm(image_paths):

        # load and pre-process image
        path = logdir + "/" + image_path + "/"
        image = np.load(path + "original_image.npy")
        if experiment_type == "ViT":
            image = np.moveaxis(image, -1, 0).astype(np.float32) / 255
        image = torch.from_numpy(image).unsqueeze(0).detach().to(device)

        if experiment_type == "reproduction":
            random_pixel_experiment = RandomPixel(
                model=model,
                image=image,
                mode="pixel",
                figure_type=figure_type,
                save_inter_masks=save_inter_masks,
            )
            random_wavelet_experiment = RandomWavelet(
                model=model,
                image=image,
                mode="wavelet",
                figure_type=figure_type,
                save_inter_masks=save_inter_masks,
            )
            pixel_rde_experiment = PixelRDE(
                mask_path=path + "exp_pixelRDE.npy",
                model=model,
                image=image,
                mode="pixel",
                figure_type=figure_type,
                save_inter_masks=save_inter_masks,
            )
            cartoonX_experiment = CartoonX_exp(
                mask_path=path + "DWTmask_cartoonX.pickle",
                model=model,
                image=image,
                mode="wavelet",
                figure_type=figure_type,
                save_inter_masks=save_inter_masks,
            )
        elif experiment_type == "ViT":
            random_pixel_experiment_ViT = RandomPixel(
                model=model_vit,
                image=image,
                mode="pixel",
                figure_type=figure_type,
                save_inter_masks=save_inter_masks,
            )
            random_wavelet_experiment_ViT = RandomWavelet(
                model=model_vit,
                image=image,
                mode="wavelet",
                figure_type=figure_type,
                save_inter_masks=save_inter_masks,
            )
            random_pixel_experiment_CNN = RandomPixel(
                model=model_cnn,
                image=image,
                mode="pixel",
                figure_type=figure_type,
                save_inter_masks=save_inter_masks,
            )
            random_wavelet_experiment_CNN = RandomWavelet(
                model=model_cnn,
                image=image,
                mode="wavelet",
                figure_type=figure_type,
                save_inter_masks=save_inter_masks,
            )
            cartoonX_CNN_experiment = CartoonX_exp(
                mask_path=path + "DWTmask_cnn.pickle",
                model=model_cnn,
                image=image,
                mode="wavelet",
                figure_type=figure_type,
                save_inter_masks=save_inter_masks,
            )
            cartoonX_ViT_experiment = CartoonX_exp(
                mask_path=path + "DWTmask_vit.pickle",
                model=model_vit,
                image=image,
                mode="wavelet",
                figure_type=figure_type,
                save_inter_masks=save_inter_masks,
            )
            attention_mask_experiment = PixelRDE(
                mask_path=path + "attention_mask.npy",
                model=model_vit,
                image=image,
                mode="pixel",
                figure_type=figure_type,
                save_inter_masks=save_inter_masks,
            )

        # Execute experiments and save distortion scores

        if experiment_type == "reproduction":
            random_pixel_scores = random_pixel_experiment.execute()
            random_pixel_scores_total.append(random_pixel_scores)

            random_wavelet_scores = random_wavelet_experiment.execute()
            random_wavelet_scores_total.append(random_wavelet_scores)

            pixel_rde_scores = pixel_rde_experiment.execute()
            pixel_rde_scores_total.append(pixel_rde_scores)

            cartoonX_scores = cartoonX_experiment.execute()
            cartoonX_scores_total.append(cartoonX_scores)
        elif experiment_type == "ViT":
            random_pixel_scores_ViT = random_pixel_experiment_ViT.execute()
            random_pixel_scores_ViT_total.append(random_pixel_scores_ViT)

            random_wavelet_scores_ViT = random_wavelet_experiment_ViT.execute()
            random_wavelet_scores_ViT_total.append(random_wavelet_scores_ViT)

            random_pixel_scores_CNN = random_pixel_experiment_CNN.execute()
            random_pixel_scores_CNN_total.append(random_pixel_scores_CNN)

            random_wavelet_scores_CNN = random_wavelet_experiment_CNN.execute()
            random_wavelet_scores_CNN_total.append(random_wavelet_scores_CNN)

            cartoonX_CNN_scores = cartoonX_CNN_experiment.execute()
            cartoonX_scores_CNN_total.append(cartoonX_CNN_scores)

            cartoonX_ViT_scores = cartoonX_ViT_experiment.execute()
            cartoonX_scores_ViT_total.append(cartoonX_ViT_scores)

            attention_mask_scores = attention_mask_experiment.execute()
            attention_masks_scores_total.append(attention_mask_scores)

    # Save data
    if experiment_type == "reproduction":
        data = {
            "random_pixel": random_pixel_scores_total,
            "random_wavelet": random_wavelet_scores_total,
            "pixel_rde": pixel_rde_scores_total,
            "cartoonX": cartoonX_scores_total,
        }
    elif experiment_type == "ViT":
        data = {
            "random_pixel_CNN": random_pixel_scores_CNN_total,
            "random_wavelet_CNN": random_wavelet_scores_CNN_total,
            "random_pixel_ViT": random_pixel_scores_ViT_total,
            "random_wavelet_ViT": random_wavelet_scores_ViT_total,
            "cartoonX_CNN": cartoonX_scores_CNN_total,
            "cartoonX_ViT": cartoonX_scores_ViT_total,
            "attention_mask": attention_masks_scores_total,
        }

    # Create storedir if it doesn't exist
    if not os.path.exists(storedir):
        os.makedirs(storedir)

    with open(
        f"{storedir}/distortion_curves_{figure_type}_l1lambda{l1lambda}.pickle", "wb"
    ) as fp:  # Save DWTmask_cartoonX
        pickle.dump(data, fp)


if __name__ == "__main__":
    # Parse CLI arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--figure_type", "-f", type=str, help="Figure type (a or b)", default="x"
    )
    parser.add_argument(
        "--logdir",
        type=str,
        default="logs/explanations/distortion_curves",
        help="logdir path",
    )
    parser.add_argument(
        "--storedir",
        type=str,
        default="logs/experiments/distortion_curves",
        help="storedir path",
    )
    parser.add_argument(
        "--save_inter_masks", type=bool, default=False, help="save intermedate masks"
    )
    parser.add_argument(
        "--experiment_type",
        type=str,
        default="reproduction",
        help="reproduction or ViT",
    )
    args = parser.parse_args()

    files = os.listdir(args.logdir)

    if args.figure_type == "x":
        distortion_curves(
            figure_type="a",
            logdir=args.logdir,
            image_paths=files,
            storedir=args.storedir,
            l1lambda="CNX20_RDE4",
            save_inter_masks=args.save_inter_masks,
            experiment_type=args.experiment_type,
        )
        distortion_curves(
            figure_type="b",
            logdir=args.logdir,
            image_paths=files,
            storedir=args.storedir,
            l1lambda="CNX20_RDE4",
            save_inter_masks=args.save_inter_masks,
            experiment_type=args.experiment_type,
        )
    else:
        distortion_curves(
            figure_type=args.figure_type,
            logdir=args.logdir,
            image_paths=files,
            storedir=args.storedir,
            l1lambda="CNX20_RDE4",
            save_inter_masks=args.save_inter_masks,
            experiment_type=args.experiment_type,
        )
