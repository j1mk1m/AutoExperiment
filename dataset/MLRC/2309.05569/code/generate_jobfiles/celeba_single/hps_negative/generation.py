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

hard_prompts_with_negation = {
    "5_o_Clock_Shadow": [
        ("a headshot of a person with 5'o clock shadow", ""),
        ("a headshot of a person", "5'o clock shadow")
    ],
    "Arched_Eyebrows": [
        ("a headshot of a person with arched eyebrows", ""),
        ("a headshot of a person", "arched eyebrows")
    ],
    "Attractive": [
        ("a headshot of an attractive person.", ""),
        ("a headshot of a person", "attractive")
    ],
    "Bags_Under_Eyes": [
        ("a headshot of a person with bags under eyes", ""),
        ("a headshot of a person", "bag under eyes")
    ],
    "Bald": [
        ("a headshot of a bald person", ""),
        ("a headshot of a person", "bald")
    ],
    "Bangs": [
        ("a headshot of a person with bangs", ""),
        ("a headshot of a person", "bangs")
    ],
    "Big_Lips": [
        ("a headshot of a person with big lips", ""),
        ("a headshot of a person with small lips", "")
    ],
    "Big_Nose": [
        ("a headshot of a person with big nose", ""),
        ("a headshot of a person with small nose", "")
    ],
    "Black_Hair": [
        ("a headshot of a person with black hair", ""),
        ("a headshot of a person", "black hair")
    ],
    "Blond_Hair": [
        ("a headshot of a person with blond hair", ""),
        ("a headshot of a person", "blond hair")
    ],
    "Blurry": [
        ("a blurry headshot of a person", ""),
        ("a sharp headshot of a person", "")
    ],
    "Brown_Hair": [
        ("a headshot of a person with brown hair", ""),
        ("a headshot of a person", "brown hair")
    ],
    "Bushy_Eyebrows": [
        ("a headshot of a person with bushy eyebrows", ""),
        ("a headshot of a person", "bushy eyebrows")
    ],
    "Chubby": [
        ("a headshot of a chubby person", ""),
        ("a headshot of a person", "chubby")
    ],
    "Double_Chin": [
        ("a headshot of a person with double chin", ""),
        ("a headshot of a person", "double chin")
    ],
    "Eyeglasses": [
        ("a headshot of a person with eyeglasses", ""),
        ("a headshot of a person", "eyeglasses")
    ],
    "Goatee": [
        ("a headshot of a person with a goatee", ""),
        ("a headshot of a person", "goatee")
    ],
    "Gray_Hair": [
        ("a headshot of a person with gray hair", ""),
        ("a headshot of a person", "gray hair")
    ],
    "Heavy_Makeup": [
        ("a headshot of a person with heavy makeup", ""),
        ("a headshot of a person", "heavy makeup")
    ],
    "High_Cheekbones": [
        ("a headshot of a person with high cheekbones", ""),
        ("a headshot of a person", "low cheekbones")
    ],
    "Male": [
        ("a headshot of a man", ""),
        ("a headshot of a woman", "")
    ],
    "Mouth_Slightly_Open": [
        ("a headshot of a person with mouth slightly open", ""),
        ("a headshot of a person with mouth closed", "")
    ],
    "Mustache": [
        ("a headshot of a person with mustache", ""),
        ("a headshot of a person", "mustache")
    ],
    "Narrow_Eyes": [
        ("a headshot of a person with narrow eyes", ""),
        ("a headshot of a person", "narrow eyes")
    ],
    "No_Beard": [
        ("a headshot of a person", "beard"),
        ("a headshot of a person", "beard")
    ],
    "Oval_Face": [
        ("a headshot of a person with oval face", ""),
        ("a headshot of a person", "oval face")
    ],
    "Pale_Skin": [
        ("a headshot of a person with pale skin", ""),
        ("a headshot of a person with dark skin", "")
    ],
    "Pointy_Nose": [
        ("a headshot of a person with pointy nose", ""),
        ("a headshot of a person", "pointy nose")
    ],
    "Receding_Hairline": [
        ("a headshot of a person with receding hairline", ""),
        ("a headshot of a person", "receding hairline")
    ],
    "Rosy_Cheeks": [
        ("a headshot of a person with rosy cheeks", ""),
        ("a headshot of a person", "rosy cheeks")
    ],
    "Sideburns": [
        ("a headshot of a person with sideburns", ""),
        ("a headshot of a person", "sideburns")
    ],
    "Smiling": [
        ("a headshot of a smiling person", ""),
        ("a headshot of a person", "smiling")
    ],
    "Straight_Hair": [
        ("a headshot of a person with straight hair", ""),
        ("a headshot of a person", "straight hair")
    ],
    "Wavy_Hair": [
        ("a headshot of a person with wavy hair", ""),
        ("a headshot of a person", "wavy hair")
    ],
    "Wearing_Earrings": [
        ("a headshot of a person wearing earrings", ""),
        ("a headshot of a person", "earrings")
    ],
    "Wearing_Hat": [
        ("a headshot of a person wearing hat", ""),
        ("a headshot of a person", "hat")
    ],
    "Wearing_Lipstick": [
        ("a headshot of a person wearing lipstick", ""),
        ("a headshot of a person", "lipstick")
    ],
    "Wearing_Necklace": [
        ("a headshot of a person wearing necklace", ""),
        ("a headshot of a person", "necklace")
    ],
    "Wearing_Necktie": [
        ("a headshot of a person wearing necktie", ""),
        ("a headshot of a person", "necktie")
    ],
    "Young": [
        ("a headshot of a young person", ""),
        ("a headshot of an old person", "")
    ],
}

seed = 0
for attribute, hard_prompts in hard_prompts_with_negation.items():
    prompt_for_positive_attribute, negative_prompt_for_positive_attribute = hard_prompts[0]
    prompt_for_negative_attribute, negative_prompt_for_negative_attribute = hard_prompts[1]

    outdir = f"results/celeba_single/hps_negative/{attribute}/{attribute}_positive"
    script = base_script.format(
        prompt=prompt_for_positive_attribute, 
        negative_prompt=negative_prompt_for_positive_attribute,
        outdir=outdir,
        seed=seed
    )
    jobfile = Path(f"jobfiles/celeba_single/hps_negative/generation/{attribute}_positive.sh")
    jobfile.parent.mkdir(exist_ok=True, parents=True)
    jobfile.write_text(script)
    seed += 1

    outdir = f"results/celeba_single/hps_negative/{attribute}/{attribute}_negative"
    script = base_script.format(
        prompt=prompt_for_negative_attribute, 
        negative_prompt=negative_prompt_for_negative_attribute,
        outdir=outdir,
        seed=seed
    )
    jobfile = Path(f"jobfiles/celeba_single/hps_negative/generation/{attribute}_negative.sh")
    jobfile.parent.mkdir(exist_ok=True, parents=True)
    jobfile.write_text(script)
    seed += 1
