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

def run(baseline="MLAgentBench", mode="FC"):
    dataset = AutoExperimentDataset(mode, os.path.join(this_path, "workspace"))
    losses = []
    logs = []
    for path, y in dataset:
        pred = run_baseline(baseline, path)
        loss = calculate_loss(pred, y, "abs")
        losses.append(loss)
        logs.append((path, y, pred))
    print(logs)
    print("Average loss: ", sum(losses) / len(losses))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=str, default="MLAgentBench", help="baseline")
    parser.add_argument("--mode", type=str, default="FC", help="FC, NC, PC-[20, 40, 60, 80]")

    args = parser.parse_args()
    print("###############################################")
    print(f"Running baseline {args.baseline} on mode {args.mode}")
    print("###############################################\n")
    run(args.baseline, args.mode)
