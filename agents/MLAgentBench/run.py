import wandb
import os
import shutil
import subprocess
from datetime import datetime
import sys

this_path = os.path.dirname(__file__)
sys.path.append(".")
from agents.MLAgentBench.MLAgentBench.runner import main

def prepare_MLAgentBench(paper_id, exp_id, mode, source):
    # Pipeline for MLAgentBench
    mla_dir = this_path 
    prompt_file = os.path.join(mla_dir, "prompts", f"prompt_{mode}.txt")
    benchmark_dir = os.path.join(mla_dir, "MLAgentBench", "benchmarks", f"{paper_id}_{exp_id}")
    if os.path.exists(benchmark_dir):
        shutil.rmtree(benchmark_dir)
    os.makedirs(benchmark_dir)
    env_dir = os.path.join(benchmark_dir, "env")
    scripts_dir = os.path.join(benchmark_dir, "scripts")
    os.mkdir(scripts_dir)

    # Copy files from source to benchmark
    shutil.copytree(os.path.join(source, "code"), env_dir, symlinks=True)
    shutil.copyfile(os.path.join(source, "experiment.txt"), os.path.join(env_dir, "experiment.txt"))
    shutil.copyfile(os.path.join(source, "paper.txt"), os.path.join(env_dir, "paper.txt"))
    shutil.copyfile(prompt_file, os.path.join(scripts_dir, "research_problem.txt"))
    
    return mla_dir 

def run_MLAgentBench(X, model, **kwargs):
    paper_id, exp_id, combined_id, mode, path = X["paper_id"], X["exp_id"], X["combined_id"], X["mode"], X["path"]
    mla_dir = prepare_MLAgentBench(paper_id, exp_id, mode, path)
    
    answer = main(X, model)
    return answer

