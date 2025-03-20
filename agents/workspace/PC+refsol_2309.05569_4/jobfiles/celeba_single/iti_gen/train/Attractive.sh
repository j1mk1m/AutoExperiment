#!/bin/bash

python train_iti_gen.py \
    --prompt="a headshot of a person" \
    --attr-list="Attractive" \
    --epochs=30 \
    --save-ckpt-per-epochs=10
