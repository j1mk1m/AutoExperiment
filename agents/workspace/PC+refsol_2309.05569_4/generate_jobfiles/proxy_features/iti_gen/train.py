from pathlib import Path
from common import attribute_lists

base_script = """
#!/bin/bash

python train_iti_gen.py \\
    --prompt="a headshot of a person" \\
    --attr-list="{attribute_list}" \\
    --epochs=30 \\
    --save-ckpt-per-epochs=10
"""
# Remove leading '\n'
base_script = base_script[1:]

for attribute_list in attribute_lists:
    underscore_separated_list = attribute_list.replace(",", "_")

    script = base_script.format(attribute_list=attribute_list)

    file = Path(f"jobfiles/proxy_features/iti_gen/train/{underscore_separated_list}.sh")
    file.parent.mkdir(exist_ok=True, parents=True)
    file.write_text(script)
