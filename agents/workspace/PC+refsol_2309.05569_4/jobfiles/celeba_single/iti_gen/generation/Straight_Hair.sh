#!/bin/bash

python generation.py \
    --config="models/sd/configs/stable-diffusion/v1-inference.yaml" \
    --ckpt="models/sd/models/ldm/stable-diffusion-v1/model.ckpt" \
    --plms \
    --attr-list="Straight_Hair" \
    --outdir="results/celeba_single/iti_gen/Straight_Hair" \
    --prompt-path="ckpts/a_headshot_of_a_person_Straight_Hair/original_prompt_embedding/basis_final_embed_29.pt" \
    --skip_grid \
    --n_iter=13 \
    --n_samples=8 \
    --seed=32
