#!/bin/sh

torchx run \
    -s local_cwd dist.ddp \
    -j 1x1 \
    --gpu 1 \
    --script src/bert4rec_main.py -- \
    --dataset_name ml-1m \
    --dataset_path ./datasets/ml-1m\
    --export_root src/outputs/pths/ \
    --lr 0.001 \
    --mask_prob 0.2 \
    --train_batch_size 8 \
    --val_batch_size 8 \
    --max_len 16 \
    --emb_dim 128 \
    --num_epochs 10 \
    --mode dmp
