from pathlib import Path

base_script = """
python evaluation.py \\
    --img-folder "{img_folder}" \\
    --class-list {class_list} 
"""
# Remove leading '\n'
base_script = base_script[1:]

class_list_by_attribute = {
    "Bald": ["a headshot of a person in bald", 
             "a headshot of a person"],
    "Male_Bald": ["a headshot of a man in bald", 
             "a headshot of a woman in bald",
             "a headshot of a man", 
             "a headshot of a woman"],
    "Mustache": ["a headshot of a person in bald", 
             "a headshot of a person"],
    "Male_Mustache": ["a headshot of a man with mustache", 
             "a headshot of a woman with mustache",
             "a headshot of a man", 
             "a headshot of a woman"],
    "No_Beard": ["a headshot of a person in bald", 
             "a headshot of a person"],
    "Male_No_Beard": ["a headshot of a man", 
             "a headshot of a woman",
             "a headshot of a man with beard", 
             "a headshot of a woman with beard"],

}

methods = [
    "iti_gen",
    "hps_negative"
]

for method in methods:
    for attribute, class_list in class_list_by_attribute.items():
        class_list = " ".join(map(lambda cls: '"' + cls + '"', class_list))
        script = base_script.format(
            img_folder=f"results/proxy_features/{method}/{attribute}",
            class_list=class_list
        )
        
        jobfile = Path(f"jobfiles/proxy_features/{method}/evaluation/{attribute}.sh")
        jobfile.parent.mkdir(exist_ok=True, parents=True)
        jobfile.write_text(script)

