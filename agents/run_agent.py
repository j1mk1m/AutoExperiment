import argparse
import wandb
import os
import shutil
import subprocess
from datetime import datetime

from agents.MLAgentBench.MLAgentBench.runner import main

this_path = os.path.dirname(__file__)

openai_api_key = open(os.path.join(this_path, "openai_api_key.txt")).read().strip()
os.environ["OPENAI_API_KEY"] = openai_api_key

def prepare_MLAgentBench(paper_id, exp_id, mode, source):
    # Pipeline for MLAgentBench
    mla_dir = os.path.join(this_path, "MLAgentBench")
    prompt_file = os.path.join(mla_dir, "prompts", f"prompt_{mode}.txt")
    benchmark_dir = os.path.join(mla_dir, "MLAgentBench", "benchmarks", "task", f"{paper_id}_{exp_id}")
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

def run_MLAgentBench(paper_id, exp_id, mode, source):
    # subprocess.run(["pip", "install", "-r", os.path.join(this_path, "MLAgentBench", "requirements.txt")])
    mla_dir = prepare_MLAgentBench(paper_id, exp_id, mode, source)
    
    # subprocess.run(["python", "-u", os.path.join(mla_dir, "MLAgentBench", "runner.py"), f"{paper_id}_{exp_id}"])
    main(f"{paper_id}_{exp_id}")

    # Retrieve final answer
    answer = 0
    submission_path = os.path.join(mla_dir, "MLAgentBench", "output.txt")
    if os.path.exists(submission_path):
        with open(submission_path, 'r') as sub_file:
            answer = sub_file.readline().strip()
    return answer

def run_agent(agent="MLAgentBench", path="../workspace"):
    dirs = path.split(os.sep)
    mode, paper_id, exp_id = dirs[-3], dirs[-2], dirs[-1]
    print(f"Running {agent} with {mode} on paper {paper_id} experiment {exp_id}")

    if agent=="MLAgentBench":
        res = run_MLAgentBench(paper_id, exp_id, mode, path)
    else:
        print(f"Agent {agent} not supported, returning 0.0")
        return 0.0

    # parse result to extract numerical value
    try:
        start = 0
        while not res[start].isdigit():
            start += 1
        end = start
        while end < len(res) and (res[end].isdigit() or res[end] == '.'):
            end += 1
        return float(res[start:end])
    except Exception as e:
        print("Could not convert to float, returning 0.0")
        return 0.0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", type=str, default="MLAgentBench", help="baseline")
    parser.add_argument("--source", type=str, default="../workspace", help="path to source directory") 

    args = parser.parse_args()
    run_agent(args.agent, args.source)


