#!/bin/bash

python models/sd/scripts/txt2img.py \
    --config="models/sd/configs/stable-diffusion/v1-inference.yaml" \
    --ckpt="models/sd/models/ldm/stable-diffusion-v1/model.ckpt" \
    --plms \
    --prompt="a headshot of a person with 5'o clock shadow" \
    --negative_prompt="" \
    --outdir="results/celeba_single/hps_negative/5_o_Clock_Shadow/5_o_Clock_Shadow_positive" \
    --skip_grid \
    --n_iter=13 \
    --n_samples=8 \
    --seed=0
