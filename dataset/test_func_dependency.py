import json
import argparse
import os
import subprocess
import shutil
import tqdm
from dataset_tmp import get_datapoint

def get_experiments(paper_id):
    with open(os.path.join("MLRC", paper_id, "experiments.jsonl"), "r") as f:
        experiments = [json.loads(line) for line in f]

    return experiments

def exp_depends_on_func(experiment, X):
    try: 
        # Run the experiment command
        process = subprocess.run(experiment["solution"], shell=True, capture_output=True, text=True, cwd=X["path"], timeout=60)
        
        # Check if NotImplementedError was raised
        if "NotImplementedError" in process.stderr:
            return True
            
        return False

    except Exception as e:
        return False


def test_func_dependency(paper_id):
    experiments = get_experiments(paper_id)

    with open(os.path.join("MLRC", paper_id, "sampled_functions.jsonl"), "r") as f:
        functions = [json.loads(line) for line in f]

    for i, func in enumerate(functions): 
        if "exp_dependencies" in func:
            continue
        func_id = func["func_id"]
        combined_id = func["paper_id"] + "_" + func_id
        print(f"Processing {combined_id}")
        X, y, metadata = get_datapoint(combined_id=combined_id)

        exp_dependencies = []
        for experiment in tqdm.tqdm(experiments):
            if exp_depends_on_func(experiment, X):
                exp_dependencies.append(experiment["exp_id"])
                print(experiment["exp_id"])
                if len(exp_dependencies) > 2:
                    break
        
        func["exp_dependencies"] = exp_dependencies
        print(f"Found {len(exp_dependencies)} experiments that depend on {combined_id}")

        shutil.rmtree(X["path"], ignore_errors=True)

        if (i+1) % 5 == 0:
            with open(os.path.join("MLRC", paper_id, "sampled_functions.jsonl"), "w") as f:
                for func in functions:
                    json.dump(func, f)
                    f.write("\n") 

    with open(os.path.join("MLRC", paper_id, "sampled_functions.jsonl"), "w") as f:
        for func in functions:
            json.dump(func, f)
            f.write("\n") 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper-id", type=str, required=True)
    args = parser.parse_args()
    test_func_dependency(args.paper_id)