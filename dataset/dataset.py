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
                    with open(os.path.join(this_dir, row["paper_id"], "functions.json"), 'r') as file:
                        contents = json.loads(file.read())
                        self.functions = contents["functions"]
                    if self.mode == "NC":
                        self.dataset.append((row, 0, self.functions))
                    else: # mode == PC, for now just one function blocked
                        self.dataset.append((row, 0, [self.functions[0]]))
                        #for i in range(len(self.functions)):
                        #    self.dataset.append((row, i, [self.functions[i]]))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        exp, index, func_to_block = self.dataset[idx]
        paper_id, exp_id = exp["paper_id"], exp["exp_id"]
        combined_id = f"{paper_id}_{exp_id}_{index}"
        if self.v: print(f"EXPERIMENT {idx}\nID {combined_id} : PaperID {paper_id} / Sub-experiment ID {exp_id}")
        return self.get_input_dict(), exp["result"]

    def get_item_by_id(self, combined_id):
        paper_id, exp_id = combined_id.split("_")
        for idx in range(len(self.dataset)):
            experiment, _, _ = self.dataset[idx]
            if experiment["paper_id"] == paper_id and experiment["exp_id"] == exp_id:
                return self.__getitem__(idx)
        return None

    def get_conda_env_by_id(self, combined_id):
        paper_id, _ = combined_id.split("_")
        for experiment, _, _ in self.dataset:
            if experiment["paper_id"] == paper_id:
                return experiment["environment"]
            
    def remove_workspace(self, X):
        path = X["path"]
        shutil.rmtree(path)
        print(f"Successfully deleted workspace {path}")
    
    def get_input_dict(self):
        workspace_dir = self.prepare_workspace()
        self.generate_ref_sol(workspace_dir)

    def generate_ref_sol(self, path):
        """ If ref_sol is included for this experiment, create ref_sol bash file """
        combined_id = self.experiment["paper_id"] + "_" + self.experiment["exp_id"]
        if os.path.isfile(os.path.join(this_dir, "refsols", combined_id + ".sh")) or "ref_sol" not in self.experiment:
            return
        with open(os.path.join(this_dir, "refsols", combined_id + ".sh"), "w") as bash_file:
            bash_file.write(f"cd {os.path.join(path, 'code')}\n")
            bash_file.write(self.experiment["ref_sol"])

    def prepare_workspace(self):
        """ Set up workspace directory for paper_id, exp_id and returns path to workspace """
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
        """Remove given function from repository and replace with NotImplementedError"""
        with open(os.path.join(workspace_code_dir, function["script"]), 'r') as file:
            lines = file.readlines()
        line = lines[function["line_start"]-1]
        num_space = 0
        while line[num_space].isspace():
            num_space += 1
        num_space = num_space + 4
        middle = ['"""'] + function["description"].split("\n") + ['"""', "raise NotImplementedError()", ""] 
        middle = [num_space * ' ' + line + "\n" for line in middle]
        content = lines[:function["line_start"]] + middle + lines[function["line_end"]+1:]
        with open(os.path.join(workspace_code_dir, function["script"]), 'w') as file:
            file.writelines(content)

    def get_meta_data(self):
        meta = {
            "experiment_description": self.experiment["description"]}
        if "refsol" in self.mode:
            if "refsol" in self.experiment:
                meta["refsol"] = self.experiment["refsol"]
            else:
                meta["refsol"] = "No reference script provided in dataset"
        return meta
