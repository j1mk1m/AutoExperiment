import os
import shutil
import csv
import json
this_dir = os.path.dirname(__file__)

class AutoExperimentDataset():
    def __init__(self, mode, experiment_csv="experiments-light.csv", workspace="../workspace", verbose=False):
        self.mode = mode
        self.workspace = workspace
        self.v = verbose

        self.dataset = []
        with open(os.path.join(this_dir, "experiment_csvs", experiment_csv), "r") as exp_file:
            reader = csv.DictReader(exp_file)
            for row in reader:
                if self.mode == "FC":
                    self.dataset.append((row, 0, []))
                else:
                    with open(os.path.join(this_dir, row["experiment_id"], "functions.json"), 'r') as file:
                        contents = json.loads(file.read())
                        self.functions = contents["functions"]
                    if self.mode == "NC":
                        self.dataset.append((row, 0, self.functions))
                    else: # mode == PC
                        for i in range(len(self.functions)):
                            self.dataset.append((row, i, [self.functions[i]]))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        exp, index, func_to_block = self.dataset[idx]
        paper_id, exp_id = exp["paper_id"], exp["exp_id"]
        loader = ExperimentLoader(self.mode, self.workspace, exp, self.v, func_to_block, index)
        if self.v:
            print(f"EXPERIMENT {idx}\nPaperID {paper_id} / Sub-experiment ID {exp_id}")
        return loader.get()

    def get_item_by_id(self, combined_id):
        paper_id, exp_id = combined_id.split("_")
        for experiment, index, func_to_block in self.dataset:
            if experiment["paper_id"] == paper_id and experiment["exp_id"] == exp_id:
                loader = ExperimentLoader(self.mode, self.workspace, experiment, self.v, func_to_block, index)
                return loader.get()
        return None

    def get_conda_env_by_id(self, combined_id):
        paper_id, _ = combined_id.split("_")
        for experiment in self.dataset:
            if experiment["paper_id"] == paper_id:
                return experiment["environment"]
            
    def remove_workspace(self, path):
        shutil.rmtree(path)
        print(f"Successfully deleted workspace {path}")

class ExperimentLoader:
    def __init__(self, mode, workspace, experiment, verbose=False, function_to_block=[], index=0):
        self.mode = mode
        self.experiment = experiment
        self.dataset_dir = this_dir
        self.workspace = workspace 
        self.v = verbose
        self.function_to_block = function_to_block 
        self.index = index

    def get(self):
        workspace_dir = self.prepare_workspace()
        self.generate_ref_sol(workspace_dir)
        return workspace_dir, float(self.experiment["result"])

    """ If ref_sol is included for this experiment, create ref_sol bash file """
    def generate_ref_sol(self, path):
        combined_id = self.experiment["paper_id"] + "_" + self.experiment["exp_id"]
        if os.path.isfile(os.path.join(this_dir, "refsols", combined_id + ".sh")) or "ref_sol" not in self.experiment:
            return
        with open(os.path.join(this_dir, "refsols", combined_id + ".sh"), "w") as bash_file:
            bash_file.write(f"cd {os.path.join(path, 'code')}\n")
            bash_file.write(self.experiment["ref_sol"])

    """ Set up workspace directory for paper_id, exp_id and returns path to workspace """
    def prepare_workspace(self):
        paper_id, exp_id = self.experiment["paper_id"], self.experiment["exp_id"]
        combined_id = paper_id + "_" + exp_id
        paper_dir = os.path.join(this_dir, paper_id)
        workspace_dir = os.path.join(self.workspace, combined_id, f"{self.mode}_{self.index}" if self.mode == "PC" else self.mode)
        if os.path.exists(workspace_dir): 
            if self.v: print(f"Using cached workspace {workspace_dir}")
            return workspace_dir
        os.makedirs(workspace_dir)

        # Copy environment.yml
        shutil.copyfile(os.path.join(paper_dir, "environment.yml"), os.path.join(workspace_dir, "environment.yml"))

        # Copy paper.txt
        shutil.copyfile(os.path.join(paper_dir, "paper.txt"), os.path.join(workspace_dir, "paper.txt"))

        # Create experiment.txt
        with open(os.path.join(workspace_dir, "experiment.txt"), 'w') as exp_file:
            exp_file.write(self.experiment["description"])

        # Set up code directory in specified mode
        source_code_dir = os.path.join(paper_dir, "code")
        workspace_code_dir = os.path.join(workspace_dir, "code")
        if self.mode == "FC":
            shutil.copytree(source_code_dir, workspace_code_dir)
        elif self.mode == "NC":
            os.mkdirs(workspace_code_dir)
        elif self.mode == "PC":
            shutil.copytree(source_code_dir, workspace_code_dir)
            for func in self.function_to_block:
                self.remove_function(workspace_code_dir, func)
            
        if self.v: print(f"Workspace {workspace_dir} prepared")
        return workspace_dir

    def remove_function(self, workspace_code_dir, function):
        before = []
        after = []
        found = False
        exited = False
        num_tabs = None
        with open(os.path.join(workspace_code_dir, function["script"]), 'r') as file:
            lines = file.readlines()
            for line in lines:
                if line.startswith("def " + function["name"]):
                    before.append(line)
                    found = True
                elif found:
                    if line.startswith("def "):
                        exited = True
                    elif num_tabs is None:
                        num_tabs = line.count("\t")
                    if exited:
                        after.append(line)
                else:
                    before.append(line)
        middle = ['"""'] + function["description"].split("\n") + ['"""', "raise NotImplementedError()"] 
        middle = [num_tabs * '\t' + line for line in middle]
        with open(os.path.join(workspace_code_dir, function["script"]), 'w') as file:
            file.writelines(before + middle + after)
