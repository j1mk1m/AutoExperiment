#!/bin/bash

python models/sd/scripts/txt2img.py \
    --config="models/sd/configs/stable-diffusion/v1-inference.yaml" \
    --ckpt="models/sd/models/ldm/stable-diffusion-v1/model.ckpt" \
    --plms \
    --prompt="a headshot of a person" \
    --negative_prompt="5'o clock shadow" \
    --outdir="results/celeba_single/hps_negative/5_o_Clock_Shadow/5_o_Clock_Shadow_negative" \
    --skip_grid \
    --n_iter=13 \
    --n_samples=8 \
    --seed=1
