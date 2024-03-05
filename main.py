import os
import argparse
from dataset.dataset import AutoExperimentDataset
from baselines.run_baseline import run_baseline
this_path = os.path.dirname(__file__) 

def calculate_loss(pred, y, loss="abs"):
    if loss == "abs":
        return abs(y - pred)
    else:
        return (y - pred)**2

def run(exp_file, baseline="MLAgentBench", mode="FC", local=False):
    dataset = AutoExperimentDataset(mode, exp_file, os.path.join(this_path, "workspace"))
    losses = []
    logs = []
    for path, y in dataset:
        pred = run_baseline(baseline, path, local)
        loss = calculate_loss(pred, y, "abs")
        losses.append(loss)
        logs.append((path, y, pred))
        dataset.remove_workspace(path)
    print(logs)
    print("Average loss: ", sum(losses) / len(losses))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_file", type=str, default="experiments-light.csv", help="name of experiment csv file")
    parser.add_argument("--baseline", type=str, default="MLAgentBench", help="baseline")
    parser.add_argument("--mode", type=str, default="FC", help="FC, NC, PC-[20, 40, 60, 80]")
    parser.add_argument("--local", action="store_true")

    args = parser.parse_args()
    print("###############################################")
    print(f"Running baseline {args.baseline} on mode {args.mode}")
    print("###############################################\n")
    run(args.exp_file, args.baseline, args.mode, args.local)
