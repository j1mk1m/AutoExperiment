#!/bin/bash

python train_iti_gen.py \
    --prompt="a headshot of a person" \
    --attr-list="Male,Eyeglasses,Mustache,Chubby,Bald,Attractive,Bangs,Smiling" \
    --epochs=30 \
    --save-ckpt-per-epochs=10
