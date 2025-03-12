import os
import shutil
import csv
import json

this_dir = os.path.dirname(__file__)


def get_datapoint(split="MLRC", mode="PC+refsol", combined_id="0000.00000_0", workspace="workspace", verbose=False, only_metadata=False, include_paper=True):
    """ Parse experiment_csv and gather experiment information """
    if "PC" in mode:
        paper_id, func_ids_string = combined_id.split("_")
        func_ids_split = func_ids_string.split(",") if func_ids_string != "" else []
        func_ids = [int(index) for index in func_ids_split]
    else:
        paper_id, exp_id = combined_id.split("_")
        func_ids_string = ""
        func_ids = []

    n = len(func_ids)

    paper_dir = os.path.join(this_dir, split, paper_id)
    if not os.path.exists(paper_dir):
        print(f"Directory does not exist {paper_dir}")
        return None, None, None

    experiment = None

    if "PC" in mode: 
        with open(os.path.join(this_dir, "MLRC", f"mlrc_n={n}.jsonl"), "r") as exp_file:
            for line in exp_file:
                row = json.loads(line)
                if row["paper_id"] == paper_id and row["func_ids"] == func_ids_string:
                    experiment = row
 
    # Get conda environment
    environment = "None"
    if split == "MLRC":
        with open(os.path.join(this_dir, "MLRC", "mlrc_main.csv"), 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row["paper_id"] == paper_id:
                    environment = row["conda_env"]

    metadata = {
        "split": split,
        "mode": mode, 
        "combined_id": combined_id, 
        "paper_id": paper_id, 
        "func_ids": func_ids,
        "environment": experiment["environment"] if "environment" in experiment else environment,
    }
    if only_metadata:
        return metadata

    workspace_dir = prepare_workspace(split, mode, combined_id, paper_id, experiment, workspace, verbose, include_paper=include_paper)
 
    X = metadata.copy()
    X["path"] = workspace_dir # path to directory containing code, paper.txt, etc
    X["experiment_description"] = experiment["experiments"] # experiment description
    X["funcs_to_block"] = experiment["func_details"] # missing function in Partial Code setting

    # If +refsol mode, create refsol.sh and include refsol in X
    X["refsol"] = create_ref_sol(experiment, workspace_dir) if "refsol" in mode else None

    # If -README mode, remove README from code base
    if "-README" in mode:
        remove_readme(X["path"])
    
    return X, experiment["results"], metadata
    

def prepare_workspace(split, mode, combined_id, paper_id, experiment, workspace, verbose, include_paper=True):
    """ Set up workspace directory for paper_id, exp_id and returns path to workspace """
    # Create a random hash string from current date/time
    # timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    # hash_str = hashlib.md5(timestamp.encode()).hexdigest()[:10]

    workspace_dir = os.path.join(workspace, mode, combined_id)
    paper_dir = os.path.join(this_dir, split, paper_id)

    if os.path.exists(workspace_dir):
        shutil.rmtree(workspace_dir)

    # Set up code directory in specified mode
    source_code_dir = os.path.join(paper_dir, "code")

    shutil.copytree(source_code_dir, workspace_dir)
        
    # Create experiment.txt
    exp_desc = experiment["experiments"]
    funcs_to_block = experiment["func_details"]
    if len(funcs_to_block) > 0:
        exp_desc += "\n\nMissing function(s): \n" + "\n".join([f"- {func['name']} in file {func['file']}" for func in funcs_to_block])
    with open(os.path.join(workspace_dir, "experiment.txt"), 'w') as exp_file:
        exp_file.write(exp_desc)

    # Copy paper.txt
    if include_paper:
        shutil.copyfile(os.path.join(paper_dir, "paper.txt"), os.path.join(workspace_dir, "paper.txt"))

    if "PC" in mode:
        funcs_to_block = remove_functions(workspace_dir, funcs_to_block)
 
    if verbose: print(f"Workspace {workspace_dir} prepared")
    return workspace_dir


""" Util functions """ 
def remove_function(workspace_code_dir, function):
    """Remove given function from repository and replace with NotImplementedError"""
    script_path = os.path.join(workspace_code_dir, function["file"])
    assert os.path.exists(script_path)
    with open(script_path, 'r') as file:
        lines = file.readlines()
    os.remove(script_path)
    
    line = lines[(function["line_start"])-1].replace("\t", "    ")
    num_space = 0
    while line[num_space].isspace():
        num_space += 1
    num_space = num_space + 4
    middle = ['"""'] + function["description"].split("\n") + ['"""', "raise NotImplementedError()", ""] 
    middle = [num_space * ' ' + line + "\n" for line in middle]
    content = lines[:(function["line_start"])] + middle + lines[(function["line_end"])+1:]
    with open(script_path, 'w') as file:
        file.writelines(content)
    function["line_end"] = function["line_start"] + len(middle)

def group_functions(functions):
    all_groups = {}
    for func in functions:
        if func["file"] not in all_groups:
            all_groups[func["file"]] = []
        all_groups[func["file"]].append(func)
    return all_groups

def remove_functions(path, functions):
    all_groups = group_functions(functions)
    new_funcs = []

    for file,funcs in all_groups.items():
        script_path = os.path.join(path, file)
        assert os.path.exists(script_path)
        with open(script_path, 'r') as file:
            original_lines = file.readlines()
        os.remove(script_path)

        # assume functions ordered based on line number
        last_line = 0
        new_lines = []

        for func in funcs:
            new_lines += original_lines[last_line:func["line_start"]-1]
            header_line = len(new_lines) + 1 + func["header_line"] - func["line_start"]
            
            header = original_lines[(func["header_line"])-1].replace("\t", "    ")
            num_space = 0
            while header[num_space].isspace():
                num_space += 1
            num_space = num_space + 4

            comments = ['"""'] + func["description"].split("\n") + ['"""', "raise NotImplementedError()", ""] 
            comments = [num_space * ' ' + line + "\n" for line in comments]
            line_start = len(new_lines) + 1
            new_lines += comments 
            line_end = len(new_lines) + 1
            last_line = func["line_end"] + 1
            new_funcs.append({"name": func["name"], "file": func["file"], "header_line": header_line, "line_start": line_start, "line_end": line_end, "relevant_paper": func['relevant_paper'], "description": func["description"]})
        
        new_lines += original_lines[last_line:]

        with open(script_path, 'w') as file:
            file.writelines(new_lines)

    return new_funcs


def remove_readme(worksapce):
    readme_path = os.path.join(worksapce, "code", "README.md")
    if os.path.exists(readme_path):
        os.remove(readme_path)
    readme_path = os.path.join(worksapce, "code", "README")
    if os.path.exists(readme_path):
        os.remove(readme_path)


def create_ref_sol(experiment, workspace):
    """ If ref_sol is included for this experiment, create refsol bash file """
    if "solution" in experiment:
        if os.path.exists(os.path.join(workspace, "refsol.sh")):
            return experiment["solution"]
        with open(os.path.join(workspace, "refsol.sh"), "w") as bash_file:
            bash_file.write(experiment["solution"])
        return experiment["solution"]
    return "No reference solution found"


def remove_workspace(path):
    shutil.rmtree(path)
    print(f"Successfully deleted workspace {path}")
