import csv
import json
import os
from itertools import combinations

this_dir = os.path.dirname(__file__)

def load_mlrc_data():
    # Load experiment data
    mlrc_exps = []
    with open('mlrc_exps.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            mlrc_exps.append(row)
            
    # Load function data
    mlrc_funcs = []
    with open('mlrc_funcs.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            mlrc_funcs.append(row)
            
    return mlrc_exps, mlrc_funcs

def generate_files(paper_ids):
    mlrc_exps, mlrc_funcs = load_mlrc_data()

    for num_removed in range(1, 6): # TODO: change this
        print(f"Num removed n = {num_removed}")
        datapoints = []

        for paper_id in paper_ids:
            print(f"Paper id: {paper_id}")
            func_details = []
            with open(os.path.join(this_dir, paper_id, "functions.jsonl"), 'r') as func_file:
                for line in func_file:
                    details = json.loads(line)
                    func_details.append(details)
            
            funcs = [func for func in mlrc_funcs if func["paper_id"] == paper_id]
            new_funcs = []
            for func in funcs:
                matched_detail = [f for f in func_details if f["func_id"] == func["func_id"]]
                if len(matched_detail) == 0:
                    print(f"None matched for {func['func_id']}")
                    continue
                detail = matched_detail[0]
                func["header_line"] = int(detail["header_line"])
                func["line_start"] = int(detail["line_start"])
                func["line_end"] = int(detail["line_end"])

                func["description"] = detail["description"]
                func["code_context"] = detail["code_context_embedding"]
                func["relevant_paper"] = detail["relevant_paper"]
                new_funcs.append(func)

            funcs = new_funcs

            for comb in combinations(funcs, num_removed):
                datapoint = {"paper_id": paper_id}
                func_ids = [f["func_id"] for f in comb]
                datapoint["func_ids"] = ",".join(func_ids)
                datapoint["func_details"] = comb

                if num_removed == 0:
                    relevant_exps = [exp for exp in mlrc_exps if exp["paper_id"] == paper_id and set(func_ids).issubset(set(exp["func_dependencies"].split(",")))]
                else:
                    relevant_exps = [exp for exp in mlrc_exps if exp["paper_id"] == paper_id and len([func_id for func_id in func_ids if func_id in exp["func_dependencies"].split(",")]) > 0]
                print(f"Function IDs: {func_ids} / number of exps: {len(relevant_exps)}")
                if len(relevant_exps) == 0:
                    continue

                experiment_string = ""
                bash_string = ""
                results = {}

                for i, exp in enumerate(relevant_exps):
                    experiment_string += f"Experiment {i+1}: " + exp["description"] + "\n"
                    bash_string += f"echo Experiment {i + 1}\n"+ exp["solution"] + "\n"
                    result = exp["result"].replace("'", "\"")
                    # print(result)
                    results[f"Experiment {i+1}"] = json.loads(result)

                experiment_string += "Return final answer as a json: {\"Experiment 1\": ..., \"Experiment 2\": ..., ...}"

                datapoint["experiments"] = experiment_string
                datapoint["solution"] = bash_string
                datapoint["results"] = results
                
                datapoints.append(datapoint)
            
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
        print(f"Average experiments for n={n}: {avg:.2f}")


paper_ids = ["2309.05569", "2303.11932", "2110.03485", "2205.00048"]
generate_files(paper_ids)

# find_averages()