import json
import os

this_dir = os.path.dirname(os.path.abspath(__file__))

def read_mlrc_n_1_jsonl(file_path="mlrc_n_1.jsonl"):
    """
    Reads the mlrc_n_1.jsonl file and yields each line as a parsed JSON object.
    """
    lines = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                lines.append(json.loads(line))
    return lines

mlrc = read_mlrc_n_1_jsonl()


dataset = []
for func in mlrc:
    paper_id = func["paper_id"]
    func_id = func["func_ids"]
    func_details = func["func_details"][0]
    path = os.path.join(this_dir, paper_id, "code", func_details["file"])
    header_line, line_start, line_end = func_details["header_line"], func_details["line_start"], func_details["line_end"]
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        func_code = "\n".join(lines[line_start-1:line_end-1])
        header = "\n".join(lines[header_line-1:line_start-1])

    dataset.append({
        "paper_id": paper_id,
        "func_id": func_id,
        "output": func_code,
        "input": header,
        "func_details": func_details,
    })

with open(os.path.join(this_dir, "mlrc_dataset.jsonl"), "w", encoding="utf-8") as f:
    for entry in dataset:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")






# Example usage:
# for entry in read_mlrc_n_1_jsonl():
#     print(entry)
