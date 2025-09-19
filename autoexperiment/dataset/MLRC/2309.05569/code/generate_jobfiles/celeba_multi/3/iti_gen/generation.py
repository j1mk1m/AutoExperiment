from pathlib import Path

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

seed = 42

attributes = [
    "Male",
    "Young",
    "Eyeglasses",
]

attr_list_comma = ",".join(attributes)
attr_list_underscore = "_".join(attributes)

prompt_path = Path(f"ckpts/a_headshot_of_a_person_{attr_list_underscore}/original_prompt_embedding/basis_final_embed_29.pt")

outdir = f"results/celeba_multi/3/iti_gen/{attr_list_underscore}"
script = base_script.format(
    attribute=attr_list_comma,
    outdir=outdir,
    prompt_path=prompt_path,
    seed=seed,
)

jobfile_path = Path(f"jobfiles/celeba_multi/3/iti_gen/generation/{attr_list_underscore}.sh")
jobfile_path.parent.mkdir(parents=True, exist_ok=True)
jobfile_path.write_text(script)