#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Union

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torchrec.distributed.model_parallel import DistributedModelParallel as DMP
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


def recalls_and_ndcgs_for_ks(
    scores: torch.Tensor, labels: torch.Tensor, ks: List[int]
) -> Dict[str, float]:
    """
    Compute Recalls and NDCGs based

    Args:
        scores (torch.Tensor) the model output tensor containing score of each item
        labels (torch.Tensor): the labels tensor
        ks (List[int]): the metrics we want to validate

    Returns:
        metrics (Dict[str, float]): The performance metrics based on given scores and labels

    """
    metrics = {}

    scores = scores
    labels = labels
    answer_count = labels.sum(1)

    labels_float = labels.float()
    _, cut = torch.sort(-scores, dim=1)
    for k in sorted(ks, reverse=True):
        cut = cut[:, :k]
        hits = labels_float.gather(1, cut)
        metrics["Recall@%d" % k] = (
            (
                hits.sum(1)
                / torch.min(torch.Tensor([k]).to(labels.device), labels.sum(1).float())
            )
            .mean()
            .cpu()
            .item()
        )

        position = torch.arange(2, 2 + k)
        # pyre-fixme[58]: `/` is not supported for operand types `int` and `Tensor`.
        weights = 1 / torch.log2(position.float())
        # pyre-fixme[16]: `float` has no attribute `to`.
        dcg = (hits * weights.to(hits.device)).sum(1)
        # pyre-fixme[16]: `float` has no attribute `to`.
        idcg = torch.Tensor([weights[: min(int(n), k)].sum() for n in answer_count]).to(
            dcg.device
        )
        ndcg = (dcg / idcg).mean()
        metrics["NDCG@%d" % k] = ndcg.cpu().item()

    return metrics

def _calculate_metrics(
    model: Union[DDP, DMP],
    batch: List[torch.LongTensor],
    metric_ks: List[int],
    device: torch.device,
) -> Dict[str, float]:
    """
    Run model on batch and calculate the metric scores (NDCG@k and Recall@k) from logits.

    Args:
        model (Union[DDP, DMP]): DMP or DDP model contains the Bert4Rec.
        batch (List[torch.Longtensor]): the data to evaluate on.
        metrics_ks: (List[int]): the metrics we want to validate.
        device (torch.device): the device to train/val/test.

    Returns:
        metrics (Dict[str, float]): the metric scores
        probs (torch.Tensor): log probabilities of each item per user.
        candidates (torch.Tensor): items per user
    """
    seqs, candidates, labels = batch
    kjt = _to_kjt(seqs, device)
    scores = model(kjt)
    scores = scores[:, -1, :]
    scores = scores.gather(1, candidates)
    log_softmax = nn.LogSoftmax(dim=-1)
    probs = log_softmax(scores)
    metrics = recalls_and_ndcgs_for_ks(scores, labels, metric_ks)

    return metrics, probs, candidates

def _dict_mean(dict_list: List[Dict[str, float]]) -> Dict[str, float]:
    mean_dict = {}
    for key in dict_list[0].keys():
        mean_dict[key] = np.mean([d[key] for d in dict_list], axis=0)
    return mean_dict

def _to_kjt(seqs: torch.LongTensor, device: torch.device) -> KeyedJaggedTensor:
    seqs_list = list(seqs)
    lengths = torch.IntTensor([value.size(0) for value in seqs_list])
    values = torch.cat(seqs_list, dim=0)

    kjt = KeyedJaggedTensor.from_lengths_sync(
        keys=["item"], values=values, lengths=lengths
    ).to(device)
    return kjt