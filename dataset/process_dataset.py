import csv
import json
import os
from itertools import combinations

this_dir = os.path.dirname(__file__)

def get_paper_ids():
    paper_ids = []
    for file in os.listdir(os.path.join(this_dir, "MLRC")):
        if os.path.isdir(os.path.join(this_dir, "MLRC", file)):
            paper_ids.append(file)
    print(paper_ids)
    return paper_ids


def load_mlrc_exps():
    # Load experiment data
    mlrc_exps = []
    with open(os.path.join(this_dir, "MLRC", "experiments.jsonl"), 'r') as f:
        for line in f:
            exp = json.loads(line)
            mlrc_exps.append(exp)
            
    return mlrc_exps


def load_mlrc_funcs(paper_ids):
    mlrc_funcs = {paper_id: [] for paper_id in paper_ids}

    functions_path = os.path.join(this_dir, "MLRC", "functions.jsonl")
    with open(functions_path, 'r') as f:
        for line in f:
            func = json.loads(line)
            mlrc_funcs[func["paper_id"]].append(func)
 
    return mlrc_funcs


def generate_files():
    paper_ids = get_paper_ids()
    mlrc_exps = load_mlrc_exps()
    mlrc_funcs = load_mlrc_funcs(paper_ids)

    for num_removed in range(0, 6):  # TODO: change this to generate files for different n
        print(f"Generating jsonl file for num removed n = {num_removed}")
        datapoints = []
        total_exps = 0

        for paper_id in paper_ids:
            # print(f"Paper id: {paper_id}")
            funcs = mlrc_funcs[paper_id]

            for comb in combinations(funcs, num_removed):
                datapoint = {"paper_id": paper_id}
                func_ids = [f["func_id"] for f in comb]
                datapoint["func_ids"] = ",".join(func_ids)
                datapoint["func_details"] = comb

                exp_dependencies = set([exp_id for func in comb for exp_id in func["exp_dependencies"]])
                if num_removed == 0:
                    relevant_exps = [exp for exp in mlrc_exps if exp["paper_id"] == paper_id]
                else:
                    relevant_exps = [exp for exp in mlrc_exps if exp["paper_id"] == paper_id and exp["exp_id"] in exp_dependencies]

                if len(relevant_exps) == 0:
                    continue
                total_exps += len(relevant_exps)

                experiment_string = ""
                bash_string = ""
                results = {}

                for i, exp in enumerate(relevant_exps):
                    experiment_string += f"Experiment {i+1}: " + exp["description"] + "\n"
                    bash_string += f"echo Experiment {i + 1}\n"+ exp["solution"] + "\n"
                    result = exp["result"].replace("'", "\"")
                    results[f"Experiment {i+1}"] = json.loads(result)

                experiment_string += "Return final answer as a json: {\"Experiment 1\": ..., \"Experiment 2\": ..., ...}"

                datapoint["experiments"] = experiment_string
                datapoint["solution"] = bash_string
                datapoint["results"] = json.dumps(results)
                
                datapoints.append(datapoint)

        print(f"Number of datapoints: {len(datapoints)}")
        print(f"average experiments: {total_exps/len(datapoints)}")
            
        # Write mlrc_funcs to jsonl file
        with open(os.path.join(this_dir, "MLRC", f"mlrc_n_{num_removed}.jsonl"), 'w') as f:
            for function in datapoints:
                json.dump(function, f)
                f.write('\n')


generate_files()