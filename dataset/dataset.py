import os
import shutil
import shutil
#import torch
#from torch.utils.data import Dataset, DataLoader
import csv
import subprocess
this_path = os.path.dirname(__file__)

class AutoExperimentDataset():
    def __init__(self, mode, experiment_csv="experiments-light.csv", workspace="../workspace"):
        self.mode = mode
        self.dataset_dir = os.path.dirname(__file__)
        self.workspace = workspace

        self.dataset = []
        with open(os.path.join(self.dataset_dir, "experiment_csvs", experiment_csv), "r") as exp_file:
            reader = csv.DictReader(exp_file)
            for row in reader:
                self.dataset.append(row)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        experiment = self.dataset[idx]
        paper_id, exp_id = experiment["paper_id"], experiment["exp_id"]
        print(f"EXPERIMENT {idx} \nPaperID {paper_id} / Sub-experiment ID {exp_id}")
        workspace_dir = self.prepare_workspace(experiment)
        self.generate_ref_sol(experiment)
        return workspace_dir, float(experiment["result"])

    def generate_ref_sol(self, experiment):
        if os.path.isfile(os.path.join(this_path, "refsols", experiment["paper_id"] + "_" + experiment["exp_id"] + ".sh")) or "ref_sol" not in experiment:
            return
        with open(os.path.join(this_path, "refsols", experiment["paper_id"] + "_" + experiment["exp_id"] + ".sh"), "w") as bash_file:
            bash_file.write("cd code\n")
            bash_file.write(experiment["ref_sol"])

    """ Set up workspace directory for paper_id, exp_id and returns path to workspace """
    def prepare_workspace(self, experiment):
        paper_dir = os.path.join(self.dataset_dir, experiment["paper_id"])
        workspace_dir = os.path.join(self.workspace, self.mode, experiment["paper_id"], experiment["exp_id"])
        if os.path.exists(workspace_dir): 
            print(f"Using cached workspace {workspace_dir}")
            return workspace_dir
        os.makedirs(workspace_dir)

        # Copy environment.yml or requirements.txt
        if os.path.isfile(os.path.join(paper_dir, "environment.yml")):
            shutil.copyfile(os.path.join(paper_dir, "environment.yml"), os.path.join(workspace_dir, "environment.yml"))
        if os.path.isfile(os.path.join(paper_dir, "requirements.txt")):
            shutil.copyfile(os.path.join(paper_dir, "requirements.txt"), os.path.join(workspace_dir, "requirements.txt"))

        # Copy paper.txt
        shutil.copyfile(os.path.join(paper_dir, "paper.txt"), os.path.join(workspace_dir, "paper.txt"))

        # Create experiment.txt
        with open(os.path.join(workspace_dir, "experiment.txt"), 'w') as exp_file:
            exp_file.write(experiment["description"])

        # Set up code directory in specified mode
        source_code_dir = os.path.join(paper_dir, "code")
        workspace_code_dir = os.path.join(workspace_dir, "code")
        if self.mode == "FC":
            shutil.copytree(source_code_dir, workspace_code_dir)
        elif self.mode == "NC":
            os.mkdirs(workspace_code_dir)

        print(f"Workspace {workspace_dir} prepared")
        return workspace_dir

    def remove_workspace(self, path):
        shutil.rmtree(path)
        print(f"Successfully deleted workspace {path}")

    def copy_code_partial(self, dataset_dir, workspace_dir, proportion=1):
        # TODO
        pass


