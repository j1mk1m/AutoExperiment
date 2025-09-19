from pathlib import Path

base_script = """
#!/bin/bash

python models/sd/scripts/txt2img.py \\
    --config="models/sd/configs/stable-diffusion/v1-inference.yaml" \\
    --ckpt="models/sd/models/ldm/stable-diffusion-v1/model.ckpt" \\
    --plms \\
    --prompt="{prompt}" \\
    --negative_prompt="{negative_prompt}" \\
    --outdir="{outdir}" \\
    --skip_grid \\
    --n_iter=13 \\
    --n_samples=8 \\
    --seed={seed}
"""
# Remove leading "\n"
base_script = base_script[1:]

hard_prompts = {
    "Male_Bald/Male_positive_Bald_positive": ("a headshot of a bald man", ""),
    "Male_Bald/Male_negative_Bald_positive": ("a headshot of a bald woman", ""),
    "Male_Bald/Male_positive_Bald_negative": ("a headshot of a man", "bald"),
    "Male_Bald/Male_negative_Bald_negative": ("a headshot of a woman", "bald"),

    "Male_Mustache/Male_positive_Mustache_positive": ("a headshot of a man with a mustache", ""),
    "Male_Mustache/Male_negative_Mustache_positive": ("a headshot of a woman with a mustache", ""),
    "Male_Mustache/Male_positive_Mustache_negative": ("a headshot of a man", "mustache"),
    "Male_Mustache/Male_negative_Mustache_negative": ("a headshot of a woman", "mustache"),

    "Male_Beard/Male_positive_Beard_positive": ("a headshot of a man with a beard", ""),
    "Male_Beard/Male_negative_Beard_positive": ("a headshot of a woman with a beard", ""),
    "Male_Beard/Male_positive_Beard_negative": ("a headshot of a man", "beard"),
    "Male_Beard/Male_negative_Beard_negative": ("a headshot of a woman", "beard"),
}

seed = 0
for attribute, (prompt, negative_prompt) in hard_prompts.items():
    outdir = f"results/proxy_features/hps_negative/{attribute}"
    script = base_script.format(
        prompt=prompt, 
        negative_prompt=negative_prompt,
        outdir=outdir,
        seed=seed
    )
    jobfile = Path(f"jobfiles/proxy_features/hps_negative/generation/{attribute}.sh")
    jobfile.parent.mkdir(exist_ok=True, parents=True)
    jobfile.write_text(script)
    seed += 1