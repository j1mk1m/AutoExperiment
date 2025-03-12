#!/bin/bash

python train_iti_gen.py \
    --prompt="a headshot of a person" \
    --attr-list="5_o_Clock_Shadow" \
    --epochs=5
