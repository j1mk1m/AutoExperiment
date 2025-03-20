from pathlib import Path
from common import attributes


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

for attribute in attributes:
    script = base_script.format(attribute=attribute)

    file = Path(f"jobfiles/celeba_single/iti_gen/train/{attribute}.sh")
    file.parent.mkdir(exist_ok=True, parents=True)
    file.write_text(script)
