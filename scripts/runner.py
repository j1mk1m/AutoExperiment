import os
import wandb
import sys
this_path = os.path.dirname(os.path.dirname(__file__))
sys.path.append(this_path)

import argparse
import shutil

from autoexperiment.dataset.dataset import get_datapoint
from autoexperiment.agents.run import run_agent
from autoexperiment.agents.run_refsol import run_refsol
from autoexperiment.agents.agent import add_agent_args 
from autoexperiment.agents.memory import add_memory_args
from autoexperiment.agents.environment import add_env_args
from autoexperiment.metrics import calculate_loss, percent_loss

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--_tags", type=str, default="auto_exp_test")
    parser.add_argument("--combined-id", type=str, default="0000.00000_0,1,2", help="combined_id = paper_id + func_ids")
    parser.add_argument("--model-engine", type=str, default="gpt-4o-mini")
    parser.add_argument("--max-agent-steps", type=int, default=50) # 50 action-taking steps
    parser.add_argument("--compute-budget", type=float, default=1.0) # $1 
    parser.add_argument("--max-compute-time", type=int, default=60*30) # 30 min

    parser.add_argument("--verbose", action="store_true")

    add_env_args(parser)
    add_agent_args(parser)
    add_memory_args(parser)

    args = parser.parse_args()

    # Initialize wandb
    tags = args._tags.split(',')
    tags.append(args.combined_id)
    tags.append(args.agent)
    tags.append(args.memory)
    tags.append(args.model_engine)
    tags.append(args.retrieval)
    wandb.init(
        project="AutoExperiment",
        entity="j1mk1m",
        tags=tags
    )
    
    # set up
    include_paper = args.retrieval != "no"
    workspace = os.path.join(os.path.dirname(this_path), "workspace")
    X, y, metadata = get_datapoint(combined_id=args.combined_id, workspace=workspace, verbose=args.verbose, include_paper=include_paper, retrieval=args.retrieval)

    if args.agent == "refsol":
        run_refsol(X)
    else:
        print("###############################")
        print(f"Agent: {args.agent}\nMemory: {args.memory}\nModel Engine: {args.model_engine}\nDatapoint: {args.combined_id}")
        print("###############################\n")

        # Run agent and get result
        pred = run_agent(args, X=X, metadata=metadata, tags=tags)
        print("###############################")
        print(f"Agent output: {str(pred)}")
        print(f"Gold output: {str(y)}")
        wandb.log({"agent_output": str(pred), "gold_output": str(y)})

        # Calculate loss
        loss_per_exp, correct_per_exp, correct_count, all_correct = calculate_loss(y, pred, percent_loss)
        print(f"Losses: {loss_per_exp}")
        print(f"Correct: {correct_per_exp}")
        print(f"All correct: {all_correct}")
        print("###############################")
        wandb.log({"losses": str(loss_per_exp), "correct": str(correct_per_exp), "correct_count": correct_count, "all_correct": all_correct})

    # clean up
    shutil.rmtree(X["path"])
