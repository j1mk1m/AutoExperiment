
#!/bin/bash

python generation.py \
    --config="models/sd/configs/stable-diffusion/v1-inference.yaml" \
    --ckpt="models/sd/models/ldm/stable-diffusion-v1/model.ckpt" \
    --plms \
    --attr-list="Skin_tone" \
    --outdir="results/reference_size/Skin_tone_50/" \
    --prompt-path="ckpts-50/a_headshot_of_a_person_Skin_tone/original_prompt_embedding/basis_final_embed_29.pt" \
    --n_iter=2 \
    --n_samples=8 \
    --skip_grid \
    --seed=42
