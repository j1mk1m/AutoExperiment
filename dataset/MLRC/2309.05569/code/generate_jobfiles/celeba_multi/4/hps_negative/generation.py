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

attr_names = [
    "Male",
    "Young",
    "Eyeglasses",
    "Smiling",
]

idx_to_bool = [
    "positive",
    "negative",
]

prompts = [
    ("a headshot of a smiling old man with eyeglasses", ""),
    ("a headshot of a smiling old man", "eyeglasses"),
    ("a headshot of an old man with eyeglasses", "smiling"),
    ("a headshot of an old man", "eyeglasses, smiling"),
    ("a headshot of a smiling old woman with eyeglasses", ""),
    ("a headshot of a smiling old woman", "eyeglasses"),
    ("a headshot of an old woman with eyeglasses", "smiling"),
    ("a headshot of an old woman", "eyeglasses, smiling"),
    ("a headshot of a smiling young man with eyeglasses", ""),
    ("a headshot of a smiling young man", "eyeglasses"),
    ("a headshot of a young man with eyeglasses", "smiling"),
    ("a headshot of a young man", "eyeglasses, smiling"),
    ("a headshot of a smiling young woman with eyeglasses", ""),
    ("a headshot of a smiling young woman", "eyeglasses"),
    ("a headshot of a young woman with eyeglasses", "smiling"),
    ("a headshot of a young woman", "eyeglasses, smiling"),
]

attr_list = "_".join(attr_names)

seed = 42
for prompt, negative_prompt in prompts:
    prompt_path = prompt.replace(" ", "_")
    idx_age = int(prompt.find("young") == -1)
    idx_gender = int(prompt.find("woman") != -1)
    idx_smile = int(prompt.find("smiling") == -1)
    idx_eyeglass = int(prompt.find("eyeglasses") == -1)

    combination = ""
    for attr, idx in zip(attr_names, [idx_gender, idx_age, idx_eyeglass, idx_smile]):
        combination += f"{attr}_{idx_to_bool[idx]}_"
    combination = combination[:-1] # Remove the trailing '_'

    outdir = f"results/celeba_multi/4/hps_negative/{attr_list}/{combination}"

    script = base_script.format(
        prompt=prompt,
        negative_prompt=negative_prompt,
        outdir=outdir,
        seed=seed,
    )

    script = base_script.format(prompt=prompt, negative_prompt=negative_prompt, outdir=outdir, seed=seed)
    jobfile = Path(f"jobfiles/celeba_multi/4/hps_negative/generation/{attr_list}/{combination}.sh")
    jobfile.parent.mkdir(exist_ok=True, parents=True)
    jobfile.write_text(script)