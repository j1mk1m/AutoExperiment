from pathlib import Path

base_script = """
python evaluation.py \\
    --img-folder "{img_folder}" \\
    --class-list {class_list} 
"""
# Remove leading '\n'
base_script = base_script[1:]

class_list_by_attribute = {
    "Male_Young_Eyeglasses": [
        'a headshot of an old man with eyeglasses', 
        'a headshot of an old man',
        'a headshot of an old woman with eyeglasses',
        'a headshot of an old woman',
        'a headshot of a young man with eyeglasses',
        'a headshot of a young man',
        'a headshot of a young woman with eyeglasses',
        'a headshot of a young woman',
    ]
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
            img_folder=f"results/celeba_multi/3/{method}/{attribute}",
            class_list=class_list
        )
        
        jobfile = Path(f"jobfiles/celeba_multi/3/{method}/evaluation/{attribute}.sh")
        jobfile.parent.mkdir(exist_ok=True, parents=True)
        jobfile.write_text(script)

