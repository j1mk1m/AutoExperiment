#!/bin/bash

python models/sd/scripts/txt2img.py \
    --config='models/sd/configs/stable-diffusion/v1-inference.yaml' \
    --ckpt='models/sd/models/ldm/stable-diffusion-v1/model.ckpt' \
    --plms \
    --prompt="a headshot of a person with small nose" \
    --outdir="results/celeba_single/hps/Big_Nose/Big_Nose_negative" \
    --skip_grid \
    --n_iter=13 \
    --n_samples=8 \
    --seed=15
