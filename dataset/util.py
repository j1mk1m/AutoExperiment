from dataset import get_datapoint
import os
this_path = os.path.dirname(__file__)

import argparse
import csv
import re
import json

main_csv = os.path.join(this_path, "MLRC", "mlrc_main.csv")
exp_csv = os.path.join(this_path, "MLRC", "mlrc_exps.csv")
func_csv = os.path.join(this_path, "MLRC", "mlrc_funcs.csv")

papers = []
with open(main_csv, mode='r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        papers.append(row)

experiments = []
with open(exp_csv, mode='r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        experiments.append(row)

functions = []
with open(func_csv, mode='r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        functions.append(row)

directories = [d for d in os.listdir(os.path.join(this_path, "MLRC")) if re.match(r'^\d+\.\d+$', d)]

for directory in directories:
    matching_experiments = [exp for exp in experiments if exp["paper_id"] == directory]
    exp_path = os.path.join(this_path, "MLRC", directory, "experiments.jsonl")
    with open(exp_path, "w") as file:
        for exp in matching_experiments:
            json.dump(exp, file)
            file.write("\n")

    matching_funcs = [func for func in functions if func["paper_id"] == directory]
    path = os.path.join(this_path, "MLRC", directory, "functions.json")
    matching_funcs = {"functions": matching_funcs}
    with open(path, "w") as file:
        json.dump(matching_funcs, file)


# for directory in directories:
#     functions_path = os.path.join(this_path, "auto_experiment", directory, "functions.json")
#     functions_jsonl_path = os.path.join(this_path, "auto_experiment", directory, "functions.jsonl")

#     if os.path.exists(functions_path):
#         with open(functions_path, 'r') as functions_file:
#             functions_data = json.load(functions_file)

#         with open(functions_jsonl_path, "w") as file:
#             for func in functions_data["functions"]:
#                 json.dump(func, file)
#                 file.write("\n")

        
