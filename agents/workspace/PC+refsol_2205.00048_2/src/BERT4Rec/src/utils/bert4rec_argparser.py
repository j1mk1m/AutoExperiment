import argparse
from typing import Any, cast, Dict, List, Union

def parse_args(argv: List[str]) -> argparse.Namespace:
        parser = argparse.ArgumentParser(description="torchrec + lightning app")

        parser.add_argument(
                "--metric_ks",
                type=int,
                nargs='+',
                default=[1, 5, 10, 20, 50, 100],
                help="k's for Precision@k and NDCG@k"
        )

        parser.add_argument(
            "--pth",
            type=str,
            default='bert4rec_model',
            help="Name of file containing trained model (without extension)"
        )

        parser.add_argument(
                "--min_user_count", 
                type=int, 
                default=5, 
                help="minimum user ratings count"
        )
        parser.add_argument(
                "--min_item_count",
                type=int,
                default=0,
                help="minimum item count for each valid user",
        )
        parser.add_argument(
                "--max_len",
                type=int,
                default=100,
                help="max length of the Bert embedding dimension",
        )
        parser.add_argument(
                "--mask_prob",
                type=float,
                default=0.15,
                help="probability of the mask",
        )
        parser.add_argument(
                "--dataset_name",
                type=str,
                default="ml-1m",
                help="dataset for experiment, current support ml-1m, ml-20m",
        )
        parser.add_argument(
                "--min_rating",
                type=int,
                default=0,
                help="minimum valid rating",
        )
        parser.add_argument(
                "--num_epochs",
                type=int,
                default=100,
                help="the number of epoch to train",
        )
        parser.add_argument(
                "--lr",
                type=float,
                default=0.001,
                help="learning rate",
        )
        parser.add_argument(
                "--decay_step",
                type=int,
                default="25",
                help="the step of weight decay",
        )
        parser.add_argument(
                "--weight_decay",
                type=float,
                default=0.0,
                help="weight decay",
        )
        parser.add_argument(
                "--gamma",
                type=float,
                default=1.0,
                help="gamma of the lr scheduler",
        )
        parser.add_argument(
                "--train_batch_size",
                type=int,
                default=128,
                help="train batch size",
        )
        parser.add_argument(
                "--val_batch_size",
                type=int,
                default=128,
                help="val batch size",
        )
        parser.add_argument(
                "--test_batch_size",
                type=int,
                default=128,
                help="test batch size",
        )
        parser.add_argument(
                "--emb_dim",
                type=int,
                default=256,
                help="dimension of the hidden layer embedding",
        )
        parser.add_argument(
                "--nhead",
                type=int,
                default=2,
                help="number of header of attention",
        )
        parser.add_argument(
                "--num_layers",
                type=int,
                default=2,
                help="number of layers of attention",
        )
        parser.add_argument(
                "--dataset_path",
                type=str,
                default=None,
                help="Path to a folder containing the dataset.",
        )
        parser.add_argument(
                "--export_root",
                type=str,
                default="",
                help="Path to save the trained model",
        )
        parser.add_argument(
                "--random_user_count",
                type=int,
                default=10,
                help="number of random users",
        )
        parser.add_argument(
                "--random_item_count",
                type=int,
                default=30,
                help="number of random items",
        )
        parser.add_argument(
                "--random_size",
                type=int,
                default=300,
                help="number of random sample size",
        )
        parser.add_argument(
                "--dupe_factor",
                type=int,
                default=3,
                help="number of duplication while generating the random masked seqs",
        )
        parser.add_argument(
                "--mode",
                type=str,
                default="dmp",
                help="dmp (distirbuted model parallel) or ddp (distributed data parallel)",
        )

        return parser.parse_args(argv)

