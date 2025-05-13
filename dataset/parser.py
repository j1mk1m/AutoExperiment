import os
import ast
import json
import random
from typing import List, Dict, Tuple

def extract_functions(file_path: str, file_name: str) -> List[Dict]:
    """
    Extract function information from a Python file using ast module.
    Returns list of dicts containing function name, file path, header line number,
    body start line number, and end line number.
    """
    functions = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()
            
        tree = ast.parse(source)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions.append({
                    'name': node.name,
                    'file': file_name,
                    'header_line': node.lineno,
                    'line_start': node.body[0].lineno,
                    'line_end': node.end_lineno
                })
                
    except Exception as e:
        print(f"Error parsing {file_path}: {str(e)}")
        
    return functions

def parse_repository(repo_path: str) -> List[Dict]:
    """
    Walk through a repository and extract function information from all Python files.
    Returns combined list of function information dictionaries.
    """
    offsets = {"2303.11932": 15, "2309.05569": 10, "2205.00048": 8, "2110.03485": 5}
    all_functions = []

    code_path = os.path.join(repo_path, "code")

    for root, _, files in os.walk(code_path):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                file_name = os.path.relpath(file_path, code_path)
                functions = extract_functions(file_path, file_name)
                all_functions.extend(functions)
    
    for i, func in enumerate(all_functions):
        func["paper_id"] = repo_path.split("/")[-1]
        func["func_id"] = str(i + offsets[repo_path.split("/")[-1]])
    
    # Save extracted functions to jsonl file in repo
    output_path = os.path.join(repo_path, "all_functions.jsonl")
    with open(output_path, 'w', encoding='utf-8') as f:
        for func in all_functions:
            json.dump(func, f)
            f.write('\n')
                
    return all_functions

def parse_mlrc():
    """
    Parse all repositories in MLRC/ directory and extract function information.
    Returns dict mapping repository names to their function information.
    """
    mlrc_path = 'MLRC'
    repo_functions = {}
    
    for repo in os.listdir(mlrc_path):
        repo_path = os.path.join(mlrc_path, repo)
        if os.path.isdir(repo_path):
            repo_functions[repo] = parse_repository(repo_path)
            
    return repo_functions


def sample_functions():
    """
    Sample a random number of functions from each repository in MLRC/ directory.
    Returns dict mapping repository names to their sampled function information.
    """
    mlrc_path = 'MLRC'
    combined_ids = ["2205.00048", "2303.11932", "2309.05569"]
    # num_functions = {"2110.03485": 45, "2205.00048": 42, "2303.11932": 38, "2309.05569": 40}
    num_functions = 50

    for repo in combined_ids:
        repo_path = os.path.join(mlrc_path, repo)
        if os.path.isdir(repo_path):
            all_functions_path = os.path.join(repo_path, "all_functions.jsonl")
            sampled_functions_path = os.path.join(repo_path, "sampled_functions.jsonl")
            with open(all_functions_path, 'r', encoding='utf-8') as f:
                repo_functions = [json.loads(line) for line in f]

            with open(sampled_functions_path, 'r', encoding='utf-8') as f:
                sampled_functions = [json.loads(line) for line in f]
                func_ids = [f["func_id"] for f in sampled_functions]
            
            filtered_functions = [func for func in repo_functions if "pytorch_wavelets" not in func["file"] and "description" in func]
            new_sampled_functions = random.sample(filtered_functions, min(num_functions, len(filtered_functions)))
            for func in new_sampled_functions:
                if func["func_id"] in func_ids:
                    continue
                sampled_functions.append(func)

            with open(sampled_functions_path, 'w', encoding='utf-8') as f:
                for func in sampled_functions:
                    json.dump(func, f)
                    f.write('\n')


if __name__ == "__main__":
    # repo_functions = parse_mlrc()
    sample_functions()
