from pathlib import Path
from common import prompt_by_attribute

base_script = """
#!/bin/bash

python train_iti_gen.py \\
    --prompt="{prompt}" \\
    --attr-list="{attribute}" \\
    --epochs=30 \\
    --save-ckpt-per-epochs=10 \\
    --refer-size-per-category={reference_size} \\
    --ckpt-path="ckpts-{reference_size}"
"""


for attribute, prompt in prompt_by_attribute.items():
    for reference_size in (10, 25, 50):
        script = base_script.format(
            prompt=prompt,
            attribute=attribute,
            reference_size=reference_size
        )
        
        jobfile = Path(f"jobfiles/reference_size/train/{attribute}_{reference_size}.sh")
        jobfile.parent.mkdir(parents=True, exist_ok=True)
        jobfile.write_text(script)