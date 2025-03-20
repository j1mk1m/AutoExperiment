import os
import wandb
this_path = os.path.dirname(__file__) 

from dataset.dataset import *

from agents.run import run_agent
from dataset.run_refsol import run_refsol

import argparse
from agents.agent import add_agent_args 
from agents.memory import add_memory_args
from agents.environment import add_env_args

def percent_loss(gold, pred):
    return abs(gold - pred) / gold

def collect_loss(gold, pred, metric_fn=percent_loss):
    loss = []
    if isinstance(pred, str):
        try:
            pred = json.loads(pred)
        except Exception as e:
            try:
                pred = int(pred)
            except Exception as e:
                print(f"Error in converting prediction: {pred}")
                return [1]

    if isinstance(pred, dict):
        for key in gold:
            if key not in pred:
                pred[key] = 0.0
            loss += collect_loss(gold[key], pred[key], metric_fn)
    elif isinstance(pred, list):
        for i in range(len(pred)):
            loss += collect_loss(gold[i], pred[i], metric_fn)
    else:
        try:
            loss += [metric_fn(gold, pred)]
        except Exception as e:
            print(f"Got error during loss calculation: {e}")
            return [1]
    return loss


def calculate_loss(gold, pred, metric_fn=percent_loss):
    try:
        if isinstance(pred, str):
            pred = json.loads(pred)

        loss_per_exp = {}
        correct_per_exp = {}
        correct_count = 0
        all_correct = True
        
        for key in gold:
            if key in pred:
                losses = collect_loss(gold[key], pred[key], metric_fn)
                loss = sum(losses) / len(losses)
            else:
                loss = 1
            loss_per_exp[key] = loss
            correct = loss <= 0.05
            if correct:
                correct_count += 1 
            correct_per_exp[key] = correct
            all_correct = all_correct and correct
        return loss_per_exp, correct_per_exp, correct_count, all_correct
    except Exception as e:
        print(e)
        return "Got parsing error", "Got parsing error", 0, False
    
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
    tags.append(args.environment)
    tags.append(args.memory)
    tags.append(args.model_engine)
    wandb.init(
        project="AutoExperiment",
        entity="j1mk1m",
        tags=tags
    )
    
    # set up
    include_paper = True
    workspace = os.path.join(this_path, "workspace")
    X, y, metadata = get_datapoint("MLRC", "PC+refsol", args.combined_id, workspace=workspace, verbose=args.verbose, include_paper=include_paper)
    wandb.log(X)
    wandb.log(metadata)

    if args.agent == "refsol":
        run_refsol(X)
    else:
        print("###############################")
        print(f"Agent: {args.agent}\nMemory: {args.memory}\nEnvironment: {args.environment}\nModel Engine: {args.model_engine}\nDatapoint: {args.combined_id}")
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
