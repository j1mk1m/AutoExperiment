#!/bin/bash

python generation.py \
    --config="models/sd/configs/stable-diffusion/v1-inference.yaml" \
    --ckpt="models/sd/models/ldm/stable-diffusion-v1/model.ckpt" \
    --plms \
    --attr-list="Pointy_Nose" \
    --outdir="results/celeba_single/iti_gen/Pointy_Nose" \
    --prompt-path="ckpts/a_headshot_of_a_person_Pointy_Nose/original_prompt_embedding/basis_final_embed_29.pt" \
    --skip_grid \
    --n_iter=13 \
    --n_samples=8 \
    --seed=27
