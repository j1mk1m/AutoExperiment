import os
import wandb
import argparse
from dataset.dataset import *
from agents.run_agent import run_agent
this_path = os.path.dirname(__file__) 

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
    parser.add_argument("--agent", type=str, default="AutoAgent", choices=["MLAgentBench", "AutoAgent", "BasicPromptAgent"], help="agent")
    parser.add_argument("--split", type=str, default="MLRC", choices=["MLRC", "super", "auto_experiment"])
    parser.add_argument("--mode", type=str, default="main", choices=["FC", "FC+refsol", "NC", "PC", "PC+refsol", "main"], help="FC, NC, PC, PC+refsol")
    parser.add_argument("--combined_id", type=str, default="0000.00000_0,1,2", help="combined_id = paper_id + func_ids")
    parser.add_argument("--retrieval", type=str, default="agent", choices=["no", "agent", "oracle"])
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.mode == "main":
        args.mode = "PC+refsol"

    # Initialize wandb
    tags = args._tags.split(',')
    tags.append(args.combined_id)
    tags.append(args.mode)
    tags.append(args.agent)
    tags.append(args.model)
    wandb.init(
        project="AutoExperiment",
        entity="j1mk1m",
        tags=tags
    )
    
    # set up
    include_paper = args.retrieval != "no"
    X, y, metadata = get_datapoint(args.split, args.mode, args.combined_id, workspace=os.path.join(this_path, "workspace"), verbose=args.verbose, include_paper=include_paper)
    wandb.log(X)
    wandb.log(metadata)
    
    if args.verbose: print(f"Running {args.agent} ({args.model}) with Mode: {args.mode} on id {args.combined_id}")

    # Run agent and get result
    pred = run_agent(agent=args.agent, X=X, metadata=metadata, model=args.model, retrieval=args.retrieval, tags=tags) 
    wandb.log({"agent_output": str(pred), "gold_output": str(y)})
    loss_per_exp, correct_per_exp, correct_count, all_correct = calculate_loss(y, pred, percent_loss)
    print(f"Losses: {loss_per_exp}")
    print(f"Correct: {all_correct}")
    wandb.log({"losses": str(loss_per_exp), "correct": str(correct_per_exp), "correct_count": correct_count, "all_correct": all_correct})

    # clean up
    shutil.rmtree(X["path"])
