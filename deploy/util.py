import csv
from itertools import combinations
import yaml
import os
import random
import json

this_dir = os.path.dirname(__file__)

num_datapoints = 20

for num_removed in range(1, 6):
    experiments = []
    with open(os.path.join(this_dir, "../dataset", "MLRC", f"mlrc_n={num_removed}.jsonl"), 'r') as f:
        for line in f:
            row = json.loads(line)
            comb_id = row["paper_id"] + "_" + row["func_ids"]
            # if "2309" in comb_id:
            experiments.append(comb_id)

    # random.shuffle(experiments)
    experiments = {"combined_id": experiments}

    with open(f'experiments_{num_removed}.yml', 'w') as f:
        yaml.dump(experiments, f)


