from pathlib import Path
from common import attributes

base_script = """
#!/bin/bash

python generation.py \\
    --config="models/sd/configs/stable-diffusion/v1-inference.yaml" \\
    --ckpt="models/sd/models/ldm/stable-diffusion-v1/model.ckpt" \\
    --plms \\
    --attr-list="{attribute}" \\
    --outdir="{outdir}" \\
    --prompt-path="{prompt_path}" \\
    --skip_grid \\
    --n_iter=13 \\
    --n_samples=8 \\
    --seed={seed}
"""
# Remove leading '\n'
base_script = base_script[1:]


for seed, attribute in enumerate(attributes):
    prompt_path = Path(f"ckpts/a_headshot_of_a_person_{attribute}/original_prompt_embedding/basis_final_embed_29.pt")

    outdir = f"results/celeba_single/iti_gen/{attribute}"
    script = base_script.format(
        attribute=attribute,
        outdir=outdir,
        prompt_path=prompt_path,
        seed=seed,
    )

    jobfile_path = Path(f"jobfiles/celeba_single/iti_gen/generation/{attribute}.sh")
    jobfile_path.parent.mkdir(parents=True, exist_ok=True)
    jobfile_path.write_text(script)
