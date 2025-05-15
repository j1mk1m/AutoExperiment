import json
import csv
import os


this_dir = os.path.dirname(__file__)

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


def get_num_func_calls(paper_id, func_details):
    file_path = os.path.join(this_dir, paper_id, "code", func_details["file"])
    with open(file_path, 'r') as f:
        lines = f.readlines()

    content = lines[func_details["line_start"]-1:func_details["line_end"]]
    content = "\n".join(content)
    count = content.count("(")
    return count

def get_metadata():
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

