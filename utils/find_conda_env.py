import argparse
import os
import sys
this_path = os.path.dirname(__file__) 
sys.path.append(os.path.join(this_path, ".."))
from dataset.dataset import *

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--combined_id", type=str, default="0000.00000_0", help="combined_id = paper_id + exp_id (e.g. paper_id = 0000.00000 and exp_id = 0 => combined_id = 0000.00000_0")
    args, _ = parser.parse_known_args()

    metadata = get_datapoint(combined_id=args.combined_id, only_metadata=True)    
    env = metadata["environment"]
    print(env)

