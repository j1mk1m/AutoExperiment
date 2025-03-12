#!/bin/bash

python models/sd/scripts/txt2img.py \
    --config="models/sd/configs/stable-diffusion/v1-inference.yaml" \
    --ckpt="models/sd/models/ldm/stable-diffusion-v1/model.ckpt" \
    --plms \
    --prompt="a headshot of a woman" \
    --negative_prompt="beard" \
    --outdir="results/proxy_features/hps_negative/Male_Beard/Male_negative_Beard_negative" \
    --skip_grid \
    --n_iter=13 \
    --n_samples=8 \
    --seed=11
