#!/bin/bash

python models/sd/scripts/txt2img.py \
    --config="models/sd/configs/stable-diffusion/v1-inference.yaml" \
    --ckpt="models/sd/models/ldm/stable-diffusion-v1/model.ckpt" \
    --plms \
    --prompt="a headshot of a person with dark skin" \
    --negative_prompt="" \
    --outdir="results/celeba_single/hps_negative/Pale_Skin/Pale_Skin_negative" \
    --skip_grid \
    --n_iter=13 \
    --n_samples=8 \
    --seed=53
