from dataset import get_datapoint, prepare_workspace
import os
this_path = os.path.dirname(__file__)

import argparse
import csv
import re
import json

directories = ["2110.03485", "2205.00048", "2303.11932", "2309.05569"]

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

