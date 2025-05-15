import json
import csv
import os


this_dir = os.path.dirname(__file__)

def get_num_func_calls(paper_id, func_details):
    file_path = os.path.join(this_dir, paper_id, "code", func_details["file"])
    with open(file_path, 'r') as f:
        lines = f.readlines()

    content = lines[func_details["line_start"]-1:func_details["line_end"]]
    content = "\n".join(content)
    count = content.count("(")
    return count

funcs = []
with open(os.path.join(this_dir, "mlrc_n=1.jsonl"), "r") as file:
    for line in file:
        funcs.append(json.loads(line))

metadata = []
for func in funcs:
    details = func["func_details"][0]
    line_count = details["line_end"] - details["line_start"] + 1
    testcase_count = len(func["results"].keys())
    func_call_count = get_num_func_calls(func["paper_id"], details)

    item = {"combined_id": f"{func['paper_id']}_{func['func_ids']}", "line_count": line_count, "testcase_count": testcase_count, "func_call_count": func_call_count}
    metadata.append(item)

# Write metadata to csv file
with open(os.path.join(this_dir, "metadata.csv"), "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=metadata[0].keys())
    writer.writeheader()
    for item in metadata:
        writer.writerow(item)

