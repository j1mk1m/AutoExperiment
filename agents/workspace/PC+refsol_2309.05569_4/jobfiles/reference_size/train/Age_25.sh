
#!/bin/bash

python train_iti_gen.py \
    --prompt="a headshot of a person" \
    --attr-list="Age" \
    --epochs=30 \
    --save-ckpt-per-epochs=10 \
    --refer-size-per-category=25 \
    --ckpt-path="ckpts-25"
