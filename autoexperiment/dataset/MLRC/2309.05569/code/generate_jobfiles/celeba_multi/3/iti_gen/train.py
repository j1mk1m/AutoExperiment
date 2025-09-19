from pathlib import Path

base_script = """
#!/bin/bash

python train_iti_gen.py \\
    --prompt="a headshot of a person" \\
    --attr-list="{attribute}" \\
    --epochs=30 \\
    --save-ckpt-per-epochs=10
"""
# Remove leading '\n'
base_script = base_script[1:]
attributes = [
    "Male",
    "Young",
    "Eyeglasses",
]

attr_list_comma = ",".join(attributes)
attr_list_underscore = "_".join(attributes)

script = base_script.format(attribute=attr_list_comma)

file = Path(f"jobfiles/celeba_multi/3/iti_gen/train/{attr_list_underscore}.sh")
file.parent.mkdir(exist_ok=True, parents=True)
file.write_text(script)
