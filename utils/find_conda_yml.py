import argparse
import os
import subprocess

this_path = os.path.dirname(__file__) 
import sys
sys.path.append(os.path.join(this_path, ".."))
from dataset.dataset import AutoExperimentDataset

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--_tags", type=str, default="auto_exp_test")
    parser.add_argument("--agent", type=str, default="MLAgentBench", help="agent")
    parser.add_argument("--mode", type=str, default="FC", choices=["FC", "NC", "PC"], help="FC, NC, PC")
    parser.add_argument("--file", type=str, default="experiments-light.csv", help="name of file to search for experiment")
    parser.add_argument("--combined_id", type=str, default="0000.00000_0", help="combined_id = paper_id + exp_id (e.g. paper_id = 0000.00000 and exp_id = 0 => combined_id = 0000.00000_0")
    parser.add_argument("--model")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    dataset = AutoExperimentDataset(args.mode, args.file, os.path.join(this_path, "..", "workspace"), args.verbose)
    path,_ = dataset.get_item_by_id(args.combined_id)
    print(os.path.abspath(os.path.join(path, "environment.yml")))
