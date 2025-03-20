#!/bin/bash

python train_iti_gen.py \
    --prompt="a headshot of a person" \
    --attr-list="Male,Young" \
    --epochs=5
