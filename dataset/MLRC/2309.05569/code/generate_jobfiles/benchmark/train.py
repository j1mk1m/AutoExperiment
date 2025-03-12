from pathlib import Path

base_script = """
#!/bin/bash

python train_iti_gen.py \\
    --prompt="a headshot of a person" \\
    --attr-list="{attribute_list}" \\
    --epochs=30 \\
    --save-ckpt-per-epochs=10
"""[1:]

attributes = [
    "Male",
    "Eyeglasses",
    "Mustache",
    "Chubby",
    "Bald",
    "Attractive",
    "Bangs",
    "Smiling"
]

for num_attributes in range(1, len(attributes) + 1):
    attribute_list = ','.join(attributes[:num_attributes])
    script = base_script.format(attribute_list=attribute_list)

    jobfile = Path(f"jobfiles/benchmark/{num_attributes}.sh")
    jobfile.parent.mkdir(parents=True, exist_ok=True)
    jobfile.write_text(script)