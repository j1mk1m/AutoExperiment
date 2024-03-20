import os
import wandb
import argparse
from dataset.dataset import AutoExperimentDataset 
from agents.run_agent import run_agent
this_path = os.path.dirname(__file__) 

def calculate_loss(pred, y, loss="abs"):
    if loss == "abs":
        return abs(y - pred)
    else:
        return (y - pred)**2 
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--_tags", type=str, default="auto_exp_test")
    parser.add_argument("--agent", type=str, default="MLAgentBench", help="agent")
    parser.add_argument("--mode", type=str, default="FC", choices=["FC", "NC", "PC"], help="FC, NC, PC")
    parser.add_argument("--file", type=str, default="experiments-light.csv", help="name of file to search for experiment")
    parser.add_argument("--combined_id", type=str, default="0000.00000_0", help="combined_id = paper_id + exp_id (e.g. paper_id = 0000.00000 and exp_id = 0 => combined_id = 0000.00000_0")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    tags = args._tags.split(',')
    tags.append(args.combined_id)
    tags.append(args.mode)
    tags.append(args.agent)
    wandb.init(
        project="auto_experiment",
        entity="j1mk1m",
        tags=tags
    )
    
    # set up
    dataset = AutoExperimentDataset(args.mode, args.file, os.path.join(this_path, "workspace"), args.verbose)
    path, y = dataset.get_item_by_id(args.combined_id)
    paper_id, exp_id = args.combined_id.split("_")
    wandb.log({"combined_id": args.combined_id, "paper_id": paper_id, "exp_id": exp_id})
    
    # Run baseline and get result
    pred = run_agent(args.agent, path) 
    loss = calculate_loss(pred, y, "abs")
    wandb.log({"agent_output": pred, "gt_output": y, "loss": loss})

    # clean up
    dataset.remove_workspace(path)
