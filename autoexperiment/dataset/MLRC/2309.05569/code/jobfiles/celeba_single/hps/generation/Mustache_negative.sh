#!/bin/bash

python models/sd/scripts/txt2img.py \
    --config='models/sd/configs/stable-diffusion/v1-inference.yaml' \
    --ckpt='models/sd/models/ldm/stable-diffusion-v1/model.ckpt' \
    --plms \
    --prompt="a headshot of a person with no mustache" \
    --outdir="results/celeba_single/hps/Mustache/Mustache_negative" \
    --skip_grid \
    --n_iter=13 \
    --n_samples=8 \
    --seed=45
