import csv
import json
import os
from itertools import combinations

this_dir = os.path.dirname(__file__)

def load_mlrc_exps():
    # Load experiment data
    mlrc_exps = []
    with open('mlrc_exps.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            mlrc_exps.append(row)
            
    return mlrc_exps

def load_mlrc_funcs(paper_ids):
    mlrc_funcs = {paper_id: [] for paper_id in paper_ids}

    for paper_id in paper_ids:
        repo_path = os.path.join(this_dir, paper_id)
        if os.path.isdir(repo_path):
            # sampled_functions_path = os.path.join(repo_path, "sampled_functions.jsonl")
            # with open(sampled_functions_path, 'r') as f:
            #     for line in f:
            #         func = json.loads(line)
            #         mlrc_funcs[paper_id].append(func)
            functions_path = os.path.join(repo_path, "functions.jsonl")
            with open(functions_path, 'r') as f:
                for line in f:
                    func = json.loads(line)
                    mlrc_funcs[paper_id].append(func)
 
    return mlrc_funcs


def generate_files(paper_ids):
    mlrc_exps = load_mlrc_exps()
    mlrc_funcs = load_mlrc_funcs(paper_ids)

    for num_removed in range(4, 6): # TODO: change this
        print(f"Num removed n = {num_removed}")
        datapoints = []
        total_exps = 0

        for paper_id in paper_ids:
            # print(f"Paper id: {paper_id}")
            funcs = mlrc_funcs[paper_id]

            # Filter functions that have no experiments
            new_funcs = []
            for func in funcs:
                if "exp_dependencies" in func and len(func["exp_dependencies"]) > 0:
                    if "code_context" not in func:
                        func["code_context"] = ""
                    if "relevant_paper" not in func:
                        func["relevant_paper"] = ""

                    new_funcs.append(func)

            funcs = new_funcs

            for comb in combinations(funcs, num_removed):
                datapoint = {"paper_id": paper_id}
                func_ids = [f["func_id"] for f in comb]
                datapoint["func_ids"] = ",".join(func_ids)
                datapoint["func_details"] = comb

                exp_dependencies = set([exp_id for func in comb for exp_id in func["exp_dependencies"]])
                relevant_exps = [exp for exp in mlrc_exps if exp["paper_id"] == paper_id and exp["exp_id"] in exp_dependencies]
                # print(f"Function IDs: {func_ids} / number of exps: {len(relevant_exps)}")
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
                datapoint["results"] = results
                
                datapoints.append(datapoint)
        print(len(datapoints))
        print(f"average experiments: {total_exps/len(datapoints)}")
            
        # Write mlrc_funcs to jsonl file
        with open(f'mlrc_n={num_removed}.jsonl', 'w') as f:
            for function in datapoints:
                json.dump(function, f)
                f.write('\n')

def find_averages():
    # Calculate average number of experiments for each n
    for n in range(6):
        total_exps = 0
        count = 0
        with open(f'mlrc_n={n}.jsonl', 'r') as f:
            for line in f:
                data = json.loads(line)
                total_exps += len(data['results'].keys())
                count += 1
        avg = total_exps / count if count > 0 else 0
        print(count)
        print(f"Average experiments for n={n}: {avg:.2f}")


paper_ids = ["2309.05569", "2303.11932", "2110.03485", "2205.00048"]
generate_files(paper_ids)

# find_averages()