from pathlib import Path
from common import prompt_by_attribute

base_script = """
#!/bin/bash

python generation.py \\
    --config="models/sd/configs/stable-diffusion/v1-inference.yaml" \\
    --ckpt="models/sd/models/ldm/stable-diffusion-v1/model.ckpt" \\
    --plms \\
    --attr-list="{attribute}" \\
    --outdir="{outdir}" \\
    --prompt-path="{prompt_path}" \\
    --n_iter=2 \\
    --n_samples=8 \\
    --skip_grid \\
    --seed=42
"""

for attribute, prompt in prompt_by_attribute.items():
    for reference_size in (10, 25, 50):
        outdir = f"results/reference_size/{attribute}_{reference_size}/"
        prompt_path = "/".join([
            f"ckpts-{reference_size}",
            prompt.replace(" ", "_") + "_" + attribute,
            "original_prompt_embedding/basis_final_embed_29.pt"
        ])
        script = base_script.format(attribute=attribute, outdir=outdir, prompt_path=prompt_path)
        
        jobfile = Path(f"jobfiles/reference_size/generation/{attribute}-{reference_size}.sh")
        jobfile.parent.mkdir(parents=True, exist_ok=True)
        jobfile.write_text(script)