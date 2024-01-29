import os
import shutil
import torch
from torch.utils.data import Dataset, DataLoader
import csv
import subprocess

# modes:
# Full code, full code(no readme), function headers, main function removed, no code

class AutoExperimentDataset(Dataset):
    def __init__(self, mode, workspace):
        self.mode = mode
        self.dataset_dir = os.path.dirname(__file__)
        self.workspace = workspace

        self.dataset = []
        with open(os.path.join(self.dataset_dir, "experiments.csv")) as exp_file:
            reader = csv.reader(exp_file, delimiter=',')
            header = True
            for row in reader:
                if header:
                    header = False
                    continue
                # print(f"Paper {row[0].strip()} experiment {row[1].strip()} result {row[3].strip()}")
                self.dataset.append({"paper_id": row[0].strip(), "exp_id": row[1].strip(), "exp_detail": row[2].strip(), "exp_result": row[3].strip()})
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        experiment = self.dataset[idx]
        paper_id, exp_id = experiment["paper_id"], experiment["exp_id"]
        print(f"EXPERIMENT {idx} \nPaperID {paper_id} / Sub-experiment ID {exp_id}")
        workspace_dir = self.prepare_workspace(experiment)
        return workspace_dir, float(experiment["exp_result"])
    
    """ Set up workspace directory for paper_id, exp_id and returns path to workspace """
    def prepare_workspace(self, experiment):
        paper_dir = os.path.join(self.dataset_dir, experiment["paper_id"])
        workspace_dir = os.path.join(self.workspace, self.mode, experiment["paper_id"], experiment["exp_id"])
        if os.path.exists(workspace_dir): 
            print(f"Using cached workspace {workspace_dir}")
            return workspace_dir
        os.makedirs(workspace_dir)

        # Copy environment.yml
        shutil.copyfile(os.path.join(paper_dir, "environment.yml"), os.path.join(workspace_dir, "environment.yml"))
        # shutil.copyfile(os.path.join(paper_dir, "requirements.txt"), os.path.join(workspace_dir, "requirements.txt"))

        # Copy paper.txt
        shutil.copyfile(os.path.join(paper_dir, "paper.txt"), os.path.join(workspace_dir, "paper.txt"))

        # Create experiment.txt
        with open(os.path.join(workspace_dir, "experiment.txt"), 'w') as exp_file:
            exp_file.write(experiment["exp_detail"])

        # Set up code directory in specified mode
        source_code_dir = os.path.join(paper_dir, "code")
        workspace_code_dir = os.path.join(workspace_dir, "code")
        if self.mode == "FC":
            shutil.copytree(source_code_dir, workspace_code_dir)
        elif self.mode == "NC":
            pass
        elif self.mode == "PC-80":
            self.copy_code_partial(source_code_dir, workspace_code_dir, 0.8)
        elif self.mode == "PC-60":
            self.copy_code_partial(source_code_dir, workspace_code_dir, 0.6)
        elif self.mode == "PC-40":
            self.copy_code_partial(source_code_dir, workspace_code_dir, 0.4)
        elif self.mode == "PC-20":
            self.copy_code_partial(source_code_dir, workspace_code_dir, 0.2)

        print(f"Workspace {workspace_dir} prepared")
        return workspace_dir

    def copy_code_partial(self, dataset_dir, workspace_dir, proportion=1):
        pass


