from pathlib import Path

base_script = """
python evaluation.py \\
    --img-folder "{img_folder}" \\
    --class-list {class_list} 
"""
# Remove leading '\n'
base_script = base_script[1:]

class_list_by_attribute = {
    "5_o_Clock_Shadow": ("a headshot of a person with 5'o clock shadow", "a headshot of a person"),
    "Arched_Eyebrows": ("a headshot of a person with arched eyebrows", "a headshot of a person"),
    "Attractive": ("a headshot of an attractive person", "a headshot of an unattractive person"),
    "Bags_Under_Eyes": ("a headshot of a person with bags under eyes", "a headshot of a person"),
    "Bald": ("a headshot of a person in bald", "a headshot of a person"),
    "Bangs": ("a headshot of a person with bangs", "a headshot of a person"),
    "Big_Lips": ("a headshot of a person with big lips", "a headshot of a person with small lips"),
    "Big_Nose": ("a headshot of a person with big nose", "a headshot of a person with small nose"),
    "Black_Hair": ("a headshot of a person with black hair", "a headshot of a person"),
    "Blond_Hair": ("a headshot of a person with blond hair", "a headshot of a person"),
    "Blurry": ("a headshot of a person in blurry", "a headshot of a person"),
    "Brown_Hair": ("a headshot of a person with brown hair", "a headshot of a person"),
    "Bushy_Eyebrows": ("a headshot of a person with bushy eyebrows", "a headshot of a person"),
    "Chubby": ("a headshot of a chubby person", "a headshot of a person"),
    "Double_Chin": ("a headshot of a person with double chin", "a headshot of a person"),
    "Eyeglasses": ("a headshot of a person with eyeglasses", "a headshot of a person"),
    "Goatee": ("a headshot of a person with goatee", "a headshot of a person"),
    "Gray_Hair": ("a headshot of a person with gray hair", "a headshot of a person"),
    "Heavy_Makeup": ("a headshot of a person with heavy makeup", "a headshot of a person"),
    "High_Cheekbones": ("a headshot of a person with high cheekbones", "a headshot of a person with low cheekbones"),
    "Male": ("a headshot of a man", "a headshot of a woman"),
    "Mouth_Slightly_Open": ("a headshot of a person with mouth slightly open", "a headshot of a person with mouth closed"),
    "Mustache": ("a headshot of a person with mustache", "a headshot of a person"),
    "Narrow_Eyes": ("a headshot of a person with narrow eyes", "a headshot of a person"),
    "No_Beard": ("a headshot of a person", "a headshot of a person with beard"),
    "Oval_Face": ("a headshot of a person with oval face", "a headshot of a person"),
    "Pale_Skin": ("a headshot of a person with pale skin", "a headshot of a person with dark skin"),
    "Pointy_Nose": ("a headshot of a person with pointy nose", "a headshot of a person"),
    "Receding_Hairline": ("a headshot of a person with receding hairline", "a headshot of a person"),
    "Rosy_Cheeks": ("a headshot of a person with rosy cheeks", "a headshot of a person"),
    "Sideburns": ("a headshot of a person with sideburns", "a headshot of a person"),
    "Smiling": ("a headshot of a person with smiling", "a headshot of a person"),
    "Straight_Hair": ("a headshot of a person with straight hair", "a headshot of a person"),
    "Wavy_Hair": ("a headshot of a person with wavy hair", "a headshot of a person"),
    "Wearing_Earrings": ("a headshot of a person wearing earrings", "a headshot of a person"),
    "Wearing_Hat": ("a headshot of a person wearing hat", "a headshot of a person"),
    "Wearing_Lipstick": ("a headshot of a person wearing lipstick", "a headshot of a person"),
    "Wearing_Necklace": ("a headshot of a person wearing necklace", "a headshot of a person"),
    "Wearing_Necktie": ("a headshot of a person wearing necktie", "a headshot of a person"),
    "Young": ("a headshot of a young person", "a headshot of an old person"),
}

methods = [
    "vanilla",
    "iti_gen",
    "hps",
    "hps_negative"
]

for method in methods:
    for attribute, class_list in class_list_by_attribute.items():
        class_list = " ".join(map(lambda cls: '"' + cls + '"', class_list))
        script = base_script.format(
            img_folder=f"results/celeba_single/{method}/{attribute}",
            class_list=class_list
        )
        
        jobfile = Path(f"jobfiles/celeba_single/{method}/evaluation/{attribute}.sh")
        jobfile.parent.mkdir(exist_ok=True, parents=True)
        jobfile.write_text(script)

