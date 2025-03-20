
#!/bin/bash

python generation.py \
    --config="models/sd/configs/stable-diffusion/v1-inference.yaml" \
    --ckpt="models/sd/models/ldm/stable-diffusion-v1/model.ckpt" \
    --plms \
    --attr-list="Colorful" \
    --outdir="results/reference_size/Colorful_25/" \
    --prompt-path="ckpts-25/a_natural_scene_Colorful/original_prompt_embedding/basis_final_embed_29.pt" \
    --n_iter=2 \
    --n_samples=8 \
    --skip_grid \
    --seed=42
