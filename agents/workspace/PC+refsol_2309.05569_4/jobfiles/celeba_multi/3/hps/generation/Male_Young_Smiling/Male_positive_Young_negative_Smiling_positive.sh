#!/bin/bash

python models/sd/scripts/txt2img.py \
    --config='models/sd/configs/stable-diffusion/v1-inference.yaml' \
    --ckpt='models/sd/models/ldm/stable-diffusion-v1/model.ckpt' \
    --plms \
    --prompt="a headshot of a smiling young man" \
    --outdir="results/celeba_multi/3/hps/Male_Young_Smiling/Male_positive_Young_negative_Smiling_positive" \
    --skip_grid \
    --n_iter=13 \
    --n_samples=8 \
    --seed=4
