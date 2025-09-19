from pathlib import Path

base_script = """
#!/bin/bash

python models/sd/scripts/txt2img.py \\
    --config='models/sd/configs/stable-diffusion/v1-inference.yaml' \\
    --ckpt='models/sd/models/ldm/stable-diffusion-v1/model.ckpt' \\
    --plms \\
    --prompt="{prompt}" \\
    --outdir="{outdir}" \\
    --skip_grid \\
    --n_iter=13 \\
    --n_samples=8 \\
    --seed={seed}
"""
# Remove leading '\n'
base_script = base_script[1:]

hard_prompts = {
    "5_o_Clock_Shadow": (
        "a headshot of a person with 5'o clock shadow", 
        "a headshot of a person with no 5'o clock shadow"
    ),
    "Arched_Eyebrows": (
        "a headshot of a person with arched eyebrows", 
        "a headshot of a person with no arched eyebrows"
    ),
    "Attractive": (
        "a headshot of an attractive person.", 
        "a headshot of an unattractive person"
    ),
    "Bags_Under_Eyes": (
        "a headshot of a person with bags under eyes", 
        "a headshot of a person with no bags under eyes"
    ),
    "Bald": (
        "a headshot of a person in bald", 
        "a headshot of a person in no bald"
    ),
    "Bangs": (
        "a headshot of a person with bangs", 
        "a headshot of a person with no bangs"
    ),
    "Big_Lips": (
        "a headshot of a person with big lips", 
        "a headshot of a person with small lips"
    ),
    "Big_Nose": (
        "a headshot of a person with big nose", 
        "a headshot of a person with small nose"
    ),
    "Black_Hair": (
        "a headshot of a person with black hair", 
        "a headshot of a person with no black hair"
    ),
    "Blond_Hair": (
        "a headshot of a person with blond hair", 
        "a headshot of a person with no blond hair"
    ),
    "Blurry": (
        "a headshot of a person in blurry", 
        "a headshot of a person in no blurry"
    ),
    "Brown_Hair": (
        "a headshot of a person with brown hair", 
        "a headshot of a person with no brown hair"
    ),
    "Bushy_Eyebrows": (
        "a headshot of a person with bushy eyebrows", 
        "a headshot of a person with no bushy eyebrows"
    ),
    "Chubby": (
        "a headshot of a chubby person", 
        "a headshot of a no chubby person"
    ),
    "Double_Chin": (
        "a headshot of a person with double chin", 
        "a headshot of a person with no double chin"
    ),
    "Eyeglasses": (
        "a headshot of a person with eyeglasses", 
        "a headshot of a person with no eyeglasses"
    ),
    "Goatee": (
        "a headshot of a person with goatee", 
        "a headshot of a person with no goatee"
    ),
    "Gray_Hair": (
        "a headshot of a person with gray hair", 
        "a headshot of a person with no gray hair"
    ),
    "Heavy_Makeup": (
        "a headshot of a person with heavy makeup", 
        "a headshot of a person with no heavy makeup"
    ),
    "High_Cheekbones": (
        "a headshot of a person with high cheekbones", 
        "a headshot of a person with low cheekbones"
    ),
    "Male": (
        "a headshot of a man", 
        "a headshot of a woman"
    ),
    "Mouth_Slightly_Open": (
        "a headshot of a person with mouth slightly open", 
        "a headshot of a person with mouth closed"
    ),
    "Mustache": (
        "a headshot of a person with mustache", 
        "a headshot of a person with no mustache"
    ),
    "Narrow_Eyes": (
        "a headshot of a person with narrow eyes", 
        "a headshot of a person with no narrow eyes"
    ),
    "No_Beard": (
        "a headshot of a person with no beard", 
        "a headshot of a person with beard"
    ),
    "Oval_Face": (
        "a headshot of a person with oval face", 
        "a headshot of a person with no oval face"
    ),
    "Pale_Skin": (
        "a headshot of a person with pale skin", 
        "a headshot of a person with dark skin"
    ),
    "Pointy_Nose": (
        "a headshot of a person with pointy nose", 
        "a headshot of a person with no pointy nose"
    ),
    "Receding_Hairline": (
        "a headshot of a person with receding hairline", 
        "a headshot of a person with no receding hairline"
    ),
    "Rosy_Cheeks": (
        "a headshot of a person with rosy cheeks", 
        "a headshot of a person with no rosy cheeks"
    ),
    "Sideburns": (
        "a headshot of a person with sideburns", 
        "a headshot of a person with no sideburns"
    ),
    "Smiling": (
        "a headshot of a person with smiling", 
        "a headshot of a person with no smiling"
    ),
    "Straight_Hair": (
        "a headshot of a person with straight hair", 
        "a headshot of a person with no straight hair"
    ),
    "Wavy_Hair": (
        "a headshot of a person with wavy hair", 
        "a headshot of a person with no wavy hair"
    ),
    "Wearing_Earrings": (
        "a headshot of a person wearing earrings", 
        "a headshot of a person without wearing earrings"
    ),
    "Wearing_Hat": (
        "a headshot of a person wearing hat", 
        "a headshot of a person without wearing hat"
    ),
    "Wearing_Lipstick": (
        "a headshot of a person wearing lipstick", 
        "a headshot of a person without wearing lipstick"
    ),
    "Wearing_Necklace": (
        "a headshot of a person wearing necklace", 
        "a headshot of a person without wearing necklace"
    ),
    "Wearing_Necktie": (
        "a headshot of a person wearing necktie", 
        "a headshot of a person without wearing necktie"
    ),
    "Young": (
        "a headshot of a young person", 
        "a headshot of an old person"
    ),
}


seed = 0
for attribute, (positive_prompt, negative_prompt) in hard_prompts.items():
    outdir = f"results/celeba_single/hps/{attribute}/{attribute}_positive"
    script = base_script.format(prompt=positive_prompt, outdir=outdir, seed=seed)
    jobfile = Path(f"jobfiles/celeba_single/hps/generation/{attribute}_positive.sh")
    jobfile.parent.mkdir(exist_ok=True, parents=True)
    jobfile.write_text(script)
    seed += 1

    outdir = f"results/celeba_single/hps/{attribute}/{attribute}_negative"
    script = base_script.format(prompt=negative_prompt, outdir=outdir, seed=seed)
    jobfile = Path(f"jobfiles/celeba_single/hps/generation/{attribute}_negative.sh")
    jobfile.parent.mkdir(exist_ok=True, parents=True)
    jobfile.write_text(script)
    seed += 1
