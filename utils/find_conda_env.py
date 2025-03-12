import argparse
import os
import sys
this_path = os.path.dirname(__file__) 
sys.path.append(os.path.join(this_path, ".."))
from dataset.dataset import *

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--_tags", type=str, default="auto_exp_test")
    parser.add_argument("--agent", type=str, default="MLAgentBench", help="agent")
    parser.add_argument("--split", type=str, default="MLRC", help="agent")
    parser.add_argument("--mode", type=str, default="FC")
    parser.add_argument("--combined_id", type=str, default="0000.00000_0", help="combined_id = paper_id + exp_id (e.g. paper_id = 0000.00000 and exp_id = 0 => combined_id = 0000.00000_0")
    parser.add_argument("--retrieval", type=str, default="agent", choices=["no", "agent", "oracle"])
    parser.add_argument("--model")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    metadata = get_datapoint(args.split, args.mode, args.combined_id, only_metadata=True)    
    env = metadata["environment"]
    print(env)

