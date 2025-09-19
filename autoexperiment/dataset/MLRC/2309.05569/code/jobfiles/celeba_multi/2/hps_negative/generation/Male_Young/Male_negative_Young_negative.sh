#!/bin/bash

python models/sd/scripts/txt2img.py \
    --config='models/sd/configs/stable-diffusion/v1-inference.yaml' \
    --ckpt='models/sd/models/ldm/stable-diffusion-v1/model.ckpt' \
    --plms \
    --prompt="a headshot of a young woman" \
    --outdir="results/celeba_multi/2/hps_negative/Male_Young/Male_negative_Young_negative" \
    --skip_grid \
    --n_iter=13 \
    --n_samples=8 \
    --seed=3
