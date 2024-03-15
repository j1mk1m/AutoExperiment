import os
import wandb
import argparse
from dataset.dataset import AutoExperimentDataset 
from baselines.run_agent import run_baseline
this_path = os.path.dirname(__file__) 

def calculate_loss(pred, y, loss="abs"):
    if loss == "abs":
        return abs(y - pred)
    else:
        return (y - pred)**2 
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--_tags", type=str, default="auto_exp_test")
    parser.add_argument("--baseline", type=str, default="MLAgentBench", help="baseline")
    parser.add_argument("--mode", type=str, default="FC", options=["FC", "NC", "PC"], help="FC, NC, PC")
    parser.add_argument("--file", type=str, default="experiments-light.csv", help="name of file to search for experiment")
    parser.add_argument("--combined_id", type=str, default="0000.00000_0", help="combined_id = paper_id + exp_id (e.g. paper_id = 0000.00000 and exp_id = 0 => combined_id = 0000.00000_0")
    args = parser.parse_args()

    wandb.init(
        project="auto_experiment",
        entity="j1mk1m",
        tags=args._tags.split(',')
    )
    
    # set up
    dataset = AutoExperimentDataset(args.mode, args.file, os.path.join(this_path, "workspace"))
    path, y = dataset.get_item_by_id(args.combined_id)
    paper_id, exp_id = args.combined_id.split("_")
    wandb.log({"combined_id": args.combined_id, "paper_id": paper_id, "exp_id": exp_id})
    
    # Run baseline and get result
    pred = run_agent(baseline, path, False) 
    loss = calculate_loss(pred, y, "abs")
    wandb.log({"agent_output": pred, "gt_output": y, "loss": loss})

    # clean up
    dataset.remove_workspace(path)
