from pathlib import Path
from common import attribute_lists

base_script = """
#!/bin/bash

python generation.py \\
    --config="models/sd/configs/stable-diffusion/v1-inference.yaml" \\
    --ckpt="models/sd/models/ldm/stable-diffusion-v1/model.ckpt" \\
    --plms \\
    --attr-list="{attribute_list}" \\
    --outdir="{outdir}" \\
    --prompt-path="{prompt_path}" \\
    --skip_grid \\
    --n_iter=13 \\
    --n_samples=8 \\
    --seed=42
"""
# Remove leading '\n'
base_script = base_script[1:]


for attribute_list in attribute_lists:
    underscore_separated_list = attribute_list.replace(",", "_")
    prompt_path = Path(f"ckpts/a_headshot_of_a_person_{underscore_separated_list}/original_prompt_embedding/basis_final_embed_29.pt")

    outdir = f"results/proxy_features/iti_gen/{underscore_separated_list}"
    script = base_script.format(
        attribute_list=attribute_list,
        outdir=outdir,
        prompt_path=prompt_path
    )

    jobfile_path = Path(f"jobfiles/proxy_features/iti_gen/generation/{underscore_separated_list}.sh")
    jobfile_path.parent.mkdir(parents=True, exist_ok=True)
    jobfile_path.write_text(script)
