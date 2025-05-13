import csv
from itertools import combinations
import yaml
import os
import random
import json

this_dir = os.path.dirname(__file__)

num_datapoints = 20

for num_removed in range(1, 4):
    with open(f'experiments_{num_removed}_full.yml', 'r') as f:
        cur_experiments = yaml.safe_load(f)["combined-id"]

    experiments = []
    with open(os.path.join(this_dir, "../dataset", "MLRC", f"mlrc_n={num_removed}_full.jsonl"), 'r') as f:
        for line in f:
            row = json.loads(line)
            comb_id = row["paper_id"] + "_" + row["func_ids"]
            if comb_id not in cur_experiments:
                experiments.append(comb_id)

    experiments = {"combined-id": experiments}

    with open(f'experiments_{num_removed}_full.yml', 'w') as f:
        yaml.dump(experiments, f)


