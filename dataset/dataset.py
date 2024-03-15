import os
import shutil
import shutil
import csv
import subprocess
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
                self.dataset.append(row)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        row = self.dataset[idx]
        paper_id, exp_id = row["paper_id"], row["exp_id"]
        loader = ExperimentLoader(self.mode, self.workspace, row, self.v)
        if self.v:
            print(f"EXPERIMENT {idx}\nPaperID {paper_id} / Sub-experiment ID {exp_id}")
        return loader.get()

    def get_item_by_id(self, combined_id):
        paper_id, exp_id = combined_id.split("_")
        for experiment in self.dataset:
            if experiment["paper_id"] == paper_id and experiment["exp_id"] == exp_id:
                loader = ExperimentLoader(self.mode, self.workspace, experiment, self.v)
                return loader.get()
        return None
            
class ExperimentLoader:
    def __init__(self, mode, workspace, experiment, verbose=False):
        self.mode = mode
        self.experiment = experiment
        self.dataset_dir = this_dir
        self.workspace = worksapce
        self.v = verbose

    def get(self):
        paper_id, exp_id = self.experiment["paper_id"], self.experiment["exp_id"]
        workspace_dir = self.prepare_workspace()
        self.generate_ref_sol()
        return workspace_dir, float(self.experiment["result"])

    """ If ref_sol is included for this experiment, create ref_sol bash file """
    def generate_ref_sol(self):
        combined_id = self.experiment["paper_id"] + "_" + self.experiment["exp_id"]
        if os.path.isfile(os.path.join(this_dir, "refsols", combined_id + ".sh")) or "ref_sol" not in self.experiment:
            return
        with open(os.path.join(this_dir, "refsols", experiment["paper_id"] + "_" + experiment["exp_id"] + ".sh"), "w") as bash_file:
            bash_file.write("cd code\n")
            bash_file.write(experiment["ref_sol"])

    """ Set up workspace directory for paper_id, exp_id and returns path to workspace """
    def prepare_workspace(self):
        paper_id, exp_id = self.experiment["paper_id"], self.experiment["exp_id"]
        paper_dir = os.path.join(this_dir, paper_id)
        workspace_dir = os.path.join(self.workspace, self.mode, paper_id, exp_id)
        if os.path.exists(workspace_dir): 
            if self.v:
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
            exp_file.write(self.experiment["description"])

        # Set up code directory in specified mode
        # TODO: Partial Code mode
        source_code_dir = os.path.join(paper_dir, "code")
        workspace_code_dir = os.path.join(workspace_dir, "code")
        if self.mode == "FC":
            shutil.copytree(source_code_dir, workspace_code_dir)
        elif self.mode == "NC":
            os.mkdirs(workspace_code_dir)

        if self.v:
            print(f"Workspace {workspace_dir} prepared")
        return workspace_dir

    def remove_workspace(self, path):
        shutil.rmtree(path)
        print(f"Successfully deleted workspace {path}")

    def copy_code_partial(self, dataset_dir, workspace_dir, proportion=1):
        # TODO
        pass




