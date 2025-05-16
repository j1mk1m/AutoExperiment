import csv
from itertools import combinations
import yaml
import os
import random
import json

this_dir = os.path.dirname(__file__)

num_datapoints = 20

for num_removed in range(1, 6):
    print(f"Generating yaml file for n={num_removed}")
    # with open(f'experiments_{num_removed}.yml', 'r') as f:
    #     cur_experiments = yaml.safe_load(f)["combined-id"]

    experiments = []
    with open(os.path.join(this_dir, "../dataset", "MLRC", f"mlrc_n_{num_removed}.jsonl"), 'r') as f:
        for line in f:
            row = json.loads(line)
            comb_id = row["paper_id"] + "_" + row["func_ids"]
            experiments.append(comb_id)

    experiments = {"combined-id": experiments}

    with open(f'experiments_{num_removed}.yml', 'w') as f:
        yaml.dump(experiments, f)


