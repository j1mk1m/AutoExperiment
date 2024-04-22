import os
import wandb
import argparse
from dataset.dataset import *
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
    parser.add_argument("--agent", type=str, default="MLAgentBench", choices=["MLAgentBench", "AutoAgent", "BasicPromptAgent"], help="agent")
    parser.add_argument("--mode", type=str, default="FC", choices=["FC", "FC+refsol", "NC", "PC", "PC+refsol"], help="FC, NC, PC, PC+refsol")
    parser.add_argument("--file", type=str, default="experiments-light.csv", help="name of file to search for experiment")
    parser.add_argument("--combined_id", type=str, default="0000.00000_0", help="combined_id = paper_id + exp_id (e.g. paper_id = 0000.00000 and exp_id = 0 => combined_id = 0000.00000_0")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo-1106")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    # Initialize wandb
    tags = args._tags.split(',')
    tags.append(args.combined_id)
    tags.append(args.mode)
    tags.append(args.agent)
    tags.append(args.model)
    wandb.init(
        project="auto_experiment",
        entity="j1mk1m",
        tags=tags
    )
    
    # set up
    X, y = get_datapoint(args.mode, args.combined_id, experiment_csv=args.file, index=args.index, workspace=os.path.join(this_path, "workspace"), verbose=args.verbose)
    paper_id, exp_id = args.combined_id.split("_")
    wandb.log(X)
    
    if args.verbose: print(f"Running {args.agent} with {args.mode} on paper {paper_id} experiment {exp_id}")

    # Run agent and get result
    pred = run_agent(agent=args.agent, X=X, model=args.model, tags=tags) 
    loss = calculate_loss(pred, y, "abs")
    wandb.log({"agent_output": pred, "result": y, "loss": loss})

    # clean up
    remove_workspace(X["path"])
