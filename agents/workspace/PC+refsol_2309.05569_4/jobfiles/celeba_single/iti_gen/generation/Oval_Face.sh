#!/bin/bash

python generation.py \
    --config="models/sd/configs/stable-diffusion/v1-inference.yaml" \
    --ckpt="models/sd/models/ldm/stable-diffusion-v1/model.ckpt" \
    --plms \
    --attr-list="Oval_Face" \
    --outdir="results/celeba_single/iti_gen/Oval_Face" \
    --prompt-path="ckpts/a_headshot_of_a_person_Oval_Face/original_prompt_embedding/basis_final_embed_29.pt" \
    --skip_grid \
    --n_iter=13 \
    --n_samples=8 \
    --seed=25
