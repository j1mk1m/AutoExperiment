import os
import shutil
import csv
import json
this_dir = os.path.dirname(__file__)

def get_datapoint(mode, combined_id, experiment_csv="experiments-light.csv", index=0, workspace="workspace", verbose=False):
    """ Parse experiment_csv and gather experiment information """
    paper_id, exp_id = combined_id.split("_")
    experiment = None
    func_to_block = None
    with open(os.path.join(this_dir, "experiment_csvs", experiment_csv), "r") as exp_file:
        reader = csv.DictReader(exp_file)
        for row in reader:
            if (row["paper_id"] == paper_id and row["exp_id"] == exp_id) or row["combined_id"] == combined_id:
                experiment = row
                if "PC" in mode: 
                    with open(os.path.join(this_dir, row["paper_id"], "functions.json"), 'r') as file:
                        contents = json.loads(file.read())
                        func_to_block = contents["functions"][index]

    workspace_dir, new_end_line = prepare_workspace(mode, experiment, index, func_to_block, workspace, verbose)
    if new_end_line:
        func_to_block["line_end"] = new_end_line

    X = {
        "mode": mode, # meta
        "combined_id": combined_id, # meta
        "paper_id": paper_id, # meta
        "exp_id": exp_id, # meta
        "index": index, # meta
        "environment": experiment["environment"] if "environment" in experiment else "None", # meta
        "path": workspace_dir, # path to directory containing code, paper.txt, etc
        "experiment_description": experiment["description"], # experiment description
        "func_to_block": func_to_block, # missing function in Partial Code setting
    } 

    # Create refsol.sh if needed
    if "refsol" in mode:
        refsol = create_ref_sol(experiment, workspace_dir)
        X["refsol"] = refsol
    
    return X, float(experiment["result"])
    
def prepare_workspace(mode, experiment, index, func_to_block, workspace, verbose):
    """ Set up workspace directory for paper_id, exp_id and returns path to workspace """
    combined_id, paper_id = experiment["combined_id"], experiment["paper_id"]
    paper_dir = os.path.join(this_dir, paper_id)
    workspace_dir = os.path.join(workspace, combined_id, f"{mode}_{index}" if "PC" in mode else mode)
    if os.path.exists(workspace_dir): 
        if verbose: print(f"Using cached workspace {workspace_dir}")
        return workspace_dir, None
    os.makedirs(workspace_dir)

    # Copy paper.txt
    shutil.copyfile(os.path.join(paper_dir, "paper.txt"), os.path.join(workspace_dir, "paper.txt"))

    # Create experiment.txt
    with open(os.path.join(workspace_dir, "experiment.txt"), 'w') as exp_file:
        exp_file.write(experiment["description"])

    # Set up code directory in specified mode
    source_code_dir = os.path.join(paper_dir, "code")
    workspace_code_dir = os.path.join(workspace_dir, "code")
    
    new_end_line = None
    if "FC" in mode:
        shutil.copytree(source_code_dir, workspace_code_dir)
    elif "PC" in mode: # Partial Code setting
        shutil.copytree(source_code_dir, workspace_code_dir)
        new_end_line = remove_function(workspace_code_dir, func_to_block)
    else:
        raise NotImplementedError()
        
    if verbose: print(f"Workspace {workspace_dir} prepared")
    return workspace_dir, new_end_line

def remove_function(workspace_code_dir, function):
    """Remove given function from repository and replace with NotImplementedError"""
    with open(os.path.join(workspace_code_dir, function["script"]), 'r') as file:
        lines = file.readlines()
    line = lines[function["line_start"]-1].replace("\t", "    ")
    num_space = 0
    while line[num_space].isspace():
        num_space += 1
    num_space = num_space + 4
    middle = ['"""'] + function["description"].split("\n") + ['"""', "raise NotImplementedError()", ""] 
    middle = [num_space * ' ' + line + "\n" for line in middle]
    content = lines[:function["line_start"]] + middle + lines[function["line_end"]+1:]
    with open(os.path.join(workspace_code_dir, function["script"]), 'w') as file:
        file.writelines(content)
    return function["line_start"] + len(middle)

def create_ref_sol(experiment, workspace):
    """ If ref_sol is included for this experiment, create refsol bash file """
    if "refsol" in experiment:
        if os.path.exists(os.path.join(workspace, "refsol.sh")):
            return experiment["refsol"]
        with open(os.path.join(workspace, "refsol.sh"), "w") as bash_file:
            bash_file.write("cd code\n")
            bash_file.write(experiment["refsol"])
        return experiment["refsol"]
    return "No reference solution found"

""" Util functions """
def get_conda_env_by_id(combined_id, experiment_csv="experiments-light.csv"):
    paper_id, _ = combined_id.split("_")
    with open(os.path.join(this_dir, "experiment_csvs", experiment_csv), "r") as exp_file:
        reader = csv.DictReader(exp_file)
        for row in reader:
            if row["paper_id"] == paper_id:
                return row["environment"]

def get_yml_by_id(combined_id):
    paper_id = combined_id.split("_")[0]
    return os.path.join(this_dir, paper_id)
            
def remove_workspace(path):
    shutil.rmtree(path)
    print(f"Successfully deleted workspace {path}")
