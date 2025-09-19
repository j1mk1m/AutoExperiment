import argparse
import yaml
import os
import random
import json

this_dir = os.path.dirname(__file__)

num_datapoints = 20

def generate_experiments_file(num_removed):
    print(f"Generating yaml file for n={num_removed}")

    experiments = []
    with open(os.path.join(this_dir, "../dataset", "MLRC", f"mlrc_n_{num_removed}.jsonl"), 'r') as f:
        for line in f:
            row = json.loads(line)
            comb_id = row["paper_id"] + "_" + row["func_ids"]
            experiments.append(comb_id)

    experiments = {"combined-id": experiments}

    with open(f'experiments_{num_removed}.yml', 'w') as f:
        yaml.dump(experiments, f)


def sample_experiments(num_removed, num_samples):
    with open(f'experiments_{num_removed}.yml', 'r') as f:
        experiments = yaml.safe_load(f)["combined-id"]

    experiments = random.sample(experiments, num_samples)

    experiments = {"combined-id": experiments}

    with open(f'experiments_{num_removed}_sampled_{num_samples}.yml', 'w') as f:
        yaml.dump(experiments, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="generate", choices=["generate", "sample"])
    parser.add_argument("--num_removed", type=int, default=1)
    parser.add_argument("--num_samples", type=int, default=100)
    args = parser.parse_args()

    if args.mode == "generate":
        generate_experiments_file(args.num_removed)
    elif args.mode == "sample":
        sample_experiments(args.num_removed, args.num_samples)
    else:
        raise ValueError(f"Invalid mode: {args.mode}")
