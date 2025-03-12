import datasets
import subprocess
import os
this_dir = os.path.dirname(__file__)

tasks = datasets.load_dataset("allenai/super", split="Expert")
for task in tasks:
    try:
        subprocess.run(["git", "clone", task["github_repo"]], check=True, cwd=os.path.join(this_dir, "super"))
    except subprocess.CalledProcessError as e:
        print(f"Failed to clone repository: {task['github_repo']}")
        print(f"Error: {e}")