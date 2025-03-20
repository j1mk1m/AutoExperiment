# TorchRec BERT4Rec

This folder contains the TorchRec implementation of [BERT4Rec](http://doi.acm.org/10.1145/3357384.3357895) (CIKM 2019), utilized per extension for the reproduction of [Joint Multisided Exposure Fairness for Recommendation](https://arxiv.org/abs/2205.00048) (SIGIR 2022). For the original TorchRec repository, refer to [here](https://github.com/pytorch/torchrec/tree/main/examples/bert4rec).



## Requirements

- Python 3.9.7
- Torch 1.13.1
- TorchX 0.3.0
- CUDA 11.6

### Download pretrained BERT4Rec

To obtain the trained models from the reproduction study, run the following command from the main `FACT` directory:
```
python download_pretrained.py
```

## How to use the code

#### Train BERT4Rec with our hyperparamters:

For training the BERT4Rec model with the same hyperparamters associated with the reproduction study, the `train.sh` bash script is provided which contains a command with the following flags:

```
torchx run \
    -s local_cwd dist.ddp \
    -j 1x1 \
    --gpu 1 \
    --script src/bert4rec_main.py -- \
    --dataset_name "ml-1m" \
    --dataset_path "./datasets/ml-1m"\
    --export_root src/outputs/pths/ \
    --lr 0.001 \
    --mask_prob 0.2 \
    --train_batch_size 8 \
    --val_batch_size 8 \
    --max_len 16 \
    --emb_dim 128 \
    --num_epochs 10 \
    --mode dmp
```

For starting the script on a single GPU, run the following command:

```
bash train.sh
```

The script will output a `.pth` model every epoch in `src/outputs/pths/`

### Evaluate pretrained BERT4Rec

To evaluate our trained BERT4Rec model, the `eval.sh` script is provided containing the following flags:

```
torchx run \
    -s local_cwd dist.ddp \
    -j 1x1 \
    --gpu 1 \
    --script src/bert4rec_evaluation.py -- \
    --dataset_name lt \
    --dataset_path src/datasets/ml-1m \
    --export_root src/outputs/pths/lt_10Epochs \
    --pth epoch_6_model.pth \
    --lr 0.001 \
    --mask_prob 0.2 \
    --train_batch_size 8 \
    --val_batch_size 8 \
    --test_batch_size 8 \
    --max_len 16 \
    --emb_dim 128 \
    --mode dmp
```

For evaluating the model on a single GPU, run the following command:

```
bash eval.sh
```



#### Train and Evaluate BERT4Rec with custom hyperparameters

To train the BERT4Rec model with custom settings, the flags in the specified `train.sh` and `eval.sh` can be adjusted. Both scripts accept the following arguments:

```
-s,	--scheduler 	scheduler to run on
-j, --j				[{min_nnodes}:]{nnodes}x{nproc_per_node}, for gpu
                    hosts, nproc_per_node must not exceed num gpus
                    (default: 1x2)
--gpu				number of gpus
--script			script or binary to run within the image (default: None)
--metric_ks			k's for Precison@k and NDCG@k
--pth				pth file of trained model
--min_user_count	minimum user ratings count
--min_item_count	minimum item count for each valid user
--max_len			max length of the Bert embedding dimension
--mask_prob			probability of the mask
--dataset_name		dataset for experiment, current support ml-1m, ml-20m, lt
--min_rating		minimum valid rating
--num_epochs		the number of epoch to train
--lr				learning rate
--decay_step		the step of weight decay
--weight_decay		weight decay
--gamma				gamma of the lr scheduler
--train_batch_size	train batch size
--val_batch_size	val batch size
--test_batch_size	test batch size
--emb_dim			dimension of the hidden layer embedding
--nhead				number of header of attention
--num_layers		number of layers of attention
--dataset_path		path to a folder containing the dataset
--export_root		path to save the trained model
--random_user_count	number of random users
--random_item_count	number of random items
--random_size		number of random sample size
--dupe_factor		number of duplication while generating the random masked seqs
--mode				distributed model parallel or distributed data parallel (dmp, ddp)
```















