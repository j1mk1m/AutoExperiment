#!/bin/bash

python train_iti_gen.py \
    --prompt="a headshot of a person" \
    --attr-list="Bushy_Eyebrows" \
    --epochs=30 \
    --save-ckpt-per-epochs=10
