from dataset import get_datapoint, prepare_workspace
import os
this_path = os.path.dirname(__file__)

import argparse
import csv
import re
import json

# parser = argparse.ArgumentParser()
# parser.add_argument("--csv", type=str, required=True, help="Path to the CSV file")
# args = parser.parse_args()

# csv_path = args.csv

# with open(csv_path, mode='r') as file:
#     reader = csv.DictReader(file)
#     for row in reader:
#         combined_id = row["combined_id"]
#         get_datapoint("FC", combined_id, workspace=os.path.join(this_path, "../", "workspace"), verbose=True)

# directories = [d for d in os.listdir(this_path) if re.match(r'^\d+\.\d+$', d)]
# print(directories)
# directories = ["2305.15933", "2403.07088", "2305.17333", "2210.14102", "2304.13148", "2401.15535"]
directories = ["2110.03485", "2203.01928", "2205.00048", "2303.11932", "2309.05569"]

for directory in directories:
    functions_path = os.path.join(this_path, "MLRC", directory, "functions.jsonl")
    if os.path.exists(functions_path):
        with open(functions_path, 'r') as functions_file:
            functions_data = []
            if "jsonl" in functions_path:
                for line in functions_file:
                    functions_data.append(json.loads(line))
            else:
                functions_data = json.load(functions_file)["functions"]
            for i, func in enumerate(functions_data):
                combined_id = f"{directory}_{i}"
                prepare_workspace("MLRC", "PC", combined_id, paper_id=directory, experiment=None, func_to_block=func, workspace=os.path.join(this_path, "../", "workspace"), verbose=True)
    else:
        print(f"functions.json doesnt exist for {directory}")

