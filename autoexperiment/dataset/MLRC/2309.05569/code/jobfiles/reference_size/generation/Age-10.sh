
#!/bin/bash

python generation.py \
    --config="models/sd/configs/stable-diffusion/v1-inference.yaml" \
    --ckpt="models/sd/models/ldm/stable-diffusion-v1/model.ckpt" \
    --plms \
    --attr-list="Age" \
    --outdir="results/reference_size/Age_10/" \
    --prompt-path="ckpts-10/a_headshot_of_a_person_Age/original_prompt_embedding/basis_final_embed_29.pt" \
    --n_iter=2 \
    --n_samples=8 \
    --skip_grid \
    --seed=42
