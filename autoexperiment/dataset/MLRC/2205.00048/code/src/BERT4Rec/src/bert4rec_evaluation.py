import torch

import sys
import json
import gzip
import torch.nn as nn

from fbgemm_gpu.split_embedding_configs import EmbOptimType
from typing import Any, cast, Dict, List
from torch import distributed as dist
from torchrec.distributed.embedding import EmbeddingCollectionSharder
from torch.nn.parallel import DistributedDataParallel as DDP
from torchrec.distributed.model_parallel import DistributedModelParallel as DMP
from torchrec.distributed.types import ModuleSharder

from models.bert4rec import BERT4Rec
from utils.bert4rec_argparser import parse_args
from utils.bert4rec_metrics import _calculate_metrics
from utils.bert4rec_movielens_datasets import Bert4RecPreprocsser, get_raw_dataframe
from utils.bert4rec_movielens_dataloader import Bert4RecDataloader


if __name__ == '__main__':
        """
        Evaluate the Bert4Rec Model on test set where each user is assigned 
        100 negative samples. 
        Will output the following files:
                metrics_log.json: performance metrics for Recall@k and NDCG@k
                run-Bert4Rec-{dataset}-fold1.txt.gz: log probs for candidate items
        Args:
                argv (List[str]): command line args.

        Returns:
                None.
        """
        args = parse_args(sys.argv[1:])
        # rank = int(os.environ["LOCAL_RANK"]) 
        rank = 0
        print("Start evaluation...")
        torch.cuda.set_device(rank)
        if torch.cuda.is_available():
                device = torch.device(f"cuda:{rank}")
                backend = "nccl"
                torch.cuda.set_device(device)
                print("GPU available:".format(torch.cuda.get_device_name(device)))
                print("device: {}".format(device))


        if not torch.distributed.is_initialized():
                print("CUDA not available")
                dist.init_process_group(backend=backend)

        print('Initializing dataloader for testset...')
        raw_data = get_raw_dataframe( 
                args.dataset_name,
                args.random_user_count,
                args.random_item_count,
                args.random_size,
                args.min_rating,
                args.dataset_path,
        )        

        df = Bert4RecPreprocsser( 
                raw_data,
                args.min_rating,
                args.min_user_count,
                args.min_item_count,
                args.dataset_name,
                args.max_len,
                args.mask_prob,
                args.dupe_factor,
        ).get_processed_dataframes()

        vocab_size = len(df["smap"]) + 2
        bert4recDataloader = Bert4RecDataloader(
                df,
                args.train_batch_size,
                args.val_batch_size,
                args.test_batch_size,
        )

        _,_, test_loader = bert4recDataloader.get_pytorch_dataloaders(0, dist.get_world_size())
        
        # Initialize BERT4Rec model on top of DMP or DDP
        model_bert4rec = BERT4Rec(
                vocab_size,
                args.max_len,
                emb_dim=args.emb_dim,
                nhead=args.nhead,
                num_layers=args.num_layers,
        ).to(device)

        fused_params: Dict[str, Any] = {}
        fused_params["optimizer"] = EmbOptimType.ADAM
        fused_params["learning_rate"] = args.lr
        fused_params["weight_decay"] = args.weight_decay
        
        if args.mode == "dmp":
                model = DMP(
                        module=model_bert4rec,
                        device=device,
                        sharders=[
                                cast(ModuleSharder[nn.Module], EmbeddingCollectionSharder(fused_params))
                        ],
                        )
                
        else:
                device_ids = [rank] if backend == "nccl" else None
                model = DDP(model_bert4rec, device_ids=device_ids)

        # Import trained BERT4Rec model
        print('Importing trained model')
        model.load_state_dict(torch.load(args.export_root + "/" + args.pth))

        # Evaluate model
        print("Evaluating model...")
        keys: List[str] = []
        for k in args.metric_ks:
                keys.append("Recall@{}".format(k))
                if k != 1:
                        keys.append("NDCG@{}".format(k))
        keys.sort()
        
        metrics_log: Dict[str, List[float]] = {key: [] for key in keys}
        scores_log: List[float] = []
        candidates_log: List[int] = []
        with torch.no_grad():
                for _,batch in enumerate(test_loader):
                        batch = [x.to(device) for x in batch]
                        metrics, scores, candidates = _calculate_metrics(model, batch, args.metric_ks, device)

                        scores_log.extend(scores.tolist())
                        candidates_log.extend(candidates.tolist())

                        for key in keys:
                                metrics_log[key].append(metrics[key])

        metrics_avg = {
                key: sum(values) / len(values) for key, values in metrics_log.items()
        }

        print("Outputting evaluation results...")
        json.dump(metrics_avg, open("{}/metrics.json".format(args.export_root), 'w'))

        if args.dataset_name == "lt":
                ds = "libraryThing"
        else:
                ds = 'ml-1M'

        with gzip.open('{}/run-Bert4Rec-{}-fold1.txt.gz'.format(args.export_root, ds), 'wb') as file_out:
                for id, user in enumerate(df["test"]["user"]):
                        for (item, score) in zip(candidates_log[id], scores_log[id]):
                                line = "\n{}\t{}\t{}\t{}\t{:.15f}\t -".format(user, 0, item, 0, score )
                                file_out.write(line.encode("utf-8"))
        file_out.close()
        print("Evaluation output saved in {}".format(args.export_root))
        

