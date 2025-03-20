#!/bin/bash

python models/sd/scripts/txt2img.py \
    --config="models/sd/configs/stable-diffusion/v1-inference.yaml" \
    --ckpt="models/sd/models/ldm/stable-diffusion-v1/model.ckpt" \
    --plms \
    --prompt="a headshot of a young woman with eyeglasses" \
    --negative_prompt="smiling" \
    --outdir="results/celeba_multi/4/hps_negative/Male_Young_Eyeglasses_Smiling/Male_negative_Young_positive_Eyeglasses_positive_Smiling_negative" \
    --skip_grid \
    --n_iter=13 \
    --n_samples=8 \
    --seed=42
