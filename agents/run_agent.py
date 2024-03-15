import argparse
import wandb
import os
import shutil
import subprocess
from datetime import datetime

this_path = os.path.dirname(__file__)
home_dir = "/home/gyeongwk"

openai_api_key = open(os.path.join(this_path, "openai_api_key.txt")).read().strip()

mla_args = [
    "--max-steps", "100",
    "--agent-max-steps", "30",
    "--max-steps-in-context", "3",
    "--max-observation-steps-in-context", "3",
    "--max-time", "72000",
    "--llm-name", "gpt-4-1106-preview",
    "--edit-script-llm-name", "gpt-4-1106-preview", 
    "--fast-llm-name", "gpt-3.5-turbo-1106"
]

def prepare_MLAgentBench(paper_id, exp_id, mode, source, log_dir):
    # Pipeline for MLAgentBench
    mla_dir = os.path.join(home_dir, "MLAgentBench")
    prompt_file = os.path.join(mla_dir, "prompts", f"prompt_{mode}.txt")
    benchmark_dir = os.path.join(mla_dir, "MLAgentBench", "benchmarks", "task")
    if os.path.exists(benchmark_dir):
        shutil.rmtree(benchmark_dir)
    os.makedirs(benchmark_dir)
    env_dir = os.path.join(benchmark_dir, "env")
    scripts_dir = os.path.join(benchmark_dir, "scripts")
    os.mkdir(scripts_dir)

    # Copy files from source to benchmark
    shutil.copyfile(os.path.join(source, "environment.yml"), os.path.join(mla_dir, "environment.yml"))
    shutil.copytree(os.path.join(source, "code"), env_dir, symlinks=True)
    shutil.copyfile(os.path.join(source, "experiment.txt"), os.path.join(env_dir, "experiment.txt"))
    shutil.copyfile(os.path.join(source, "paper.txt"), os.path.join(env_dir, "paper.txt"))
    shutil.copyfile(prompt_file, os.path.join(scripts_dir, "research_problem.txt"))
    
    # Create new log directory
    identifier = f"{mode}_{paper_id}_{exp_id}".lower()
    log_path = os.path.join(log_dir, identifier)
    os.mkdir(log_path)

    return mla_dir, log_path

def run_MLAgentBench(paper_id, exp_id, mode, source, local, log_dir):
    mla_dir, log_path = prepare_MLAgentBench(paper_id, exp_id, mode, source, log_dir)
    
    if local: # run locally
        subprocess.run(["python", "-u", os.path.join(mla_dir, "MLAgentBench", "runner.py"), "--task", "task", "--log-dir", log_path, "--work-dir", os.path.join(mla_dir, "workspace")] + mla_args)
    else:
        ### Docker ###
        print("Set up done. Proceeding to create docker image and run...")
        name = f"mla_{mode}_{paper_id}_{exp_id}".lower()
        subprocess.run(["docker", "run", "--name", name, "--gpus", "all", "--shm-size=2g", "-e", f"OPENAI_API_KEY={openai_api_key}","-v", f"{mla_dir}:/app/tmp:ro", "base_image"])
        # After container stops, copy output and log to local
        subprocess.run(["docker", "cp", f"{name}:/app/MLAgentBench/MLAgentBench/output.txt", os.path.join(log_path, "output.txt")])
        subprocess.run(["docker", "cp", f"{name}:/app/MLAgentBench/logs/agent_log/", log_path])
        subprocess.run(["docker", "rm", name])

    # Retrieve final answer
    answer = 0
    submission_path = os.path.join(log_path, "output.txt")
    if os.path.exists(submission_path):
        with open(submission_path, 'r') as sub_file:
            answer = sub_file.readline().strip()
    return answer

def run_agent(agent="MLAgentBench", path="../workspace", local=False, log_dir=os.path.join(this_path, "logs", "tmp")):
    dirs = path.split(os.sep)
    mode, paper_id, exp_id = dirs[-3], dirs[-2], dirs[-1]
    print(f"Running {agent} with {mode} on paper {paper_id} experiment {exp_id}")

    if agent=="MLAgentBench":
        res = run_MLAgentBench(paper_id, exp_id, mode, path, local, log_dir)
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
    parser.add_argument("--local", action="store_true")

    args = parser.parse_args()
    run_agent(args.agent, args.source, args.local)


