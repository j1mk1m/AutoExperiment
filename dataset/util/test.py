import os
import json

def read_functions(paper_ids):
    # Get list of paper IDs by looking at subdirectories in MLRC folder
    this_dir = os.path.dirname(__file__)

    # Read functions.jsonl for each paper
    for paper_id in paper_ids:
        functions_path = os.path.join(this_dir, paper_id, "functions.jsonl")
        functions = []
        with open(functions_path, 'r') as f:
            for line in f:
                functions.append(json.loads(line))
        
        for func in functions:
            func["header_line"] = int(func["header_line"])
            func["line_start"] = int(func["line_start"])
            func["line_end"] = int(func["line_end"])

        with open(functions_path, 'w') as f:
            for func in functions:
                json.dump(func, f)
                f.write("\n")
            

if __name__ == "__main__":
    paper_ids = ["2309.05569", "2303.11932", "2110.03485", "2205.00048"]
    functions = read_functions(paper_ids)
    