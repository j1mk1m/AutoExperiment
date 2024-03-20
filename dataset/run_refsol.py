import os
import shutil
import subprocess
import argparse
import time
import sys

this_path = os.path.dirname(__file__)
sys.path.append(os.path.join(this_path, ".."))
from dataset.dataset import AutoExperimentDataset

def run_refsol(paper_id, exp_id, mode, path):
    refsol = os.path.abspath(os.path.join(this_path, "refsols", f"{paper_id}_{exp_id}.sh"))
    subprocess.run(["bash", refsol])

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="experiments.csv", help="experiment file name")
    parser.add_argument("--workspace", default="/home/gyeongwk/workspace")
    args = parser.parse_args()

    dataset = AutoExperimentDataset("FC", args.file, args.workspace)
 
    for path, y in dataset:
        start = time.time()
        head, exp_id = os.path.split(path)
        _, paper_id = os.path.split(head)
        refsol = os.path.join(this_path, "refsols", f"{paper_id}_{exp_id}.sh")
        shutil.copyfile(refsol, os.path.join(path, "refsol.sh"))

        print("Running docker container...")

        name = f"refsol_{paper_id}_{exp_id}".lower()
        subprocess.run(["sudo", "docker", "run", "--rm", "--name", name, "--shm-size=2g", "--gpus", "all", "-v", f"{path}:/app/tmp:ro", "refsol_image"])

        dataset.remove_workspace(path)
        end = time.time()
        print(f"Run time: {end - start} seconds\n\n")
