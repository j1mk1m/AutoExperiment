import argparse
import os
import shutil
import subprocess
from datetime import datetime

this_path = os.path.dirname(__file__)
home_dir = "/home/gyeongwk"

def run_MLAgentBench(paper_id, exp_id, mode, source):
    # Pipeline for MLAgentBench
    mla_dir = os.path.join(home_dir, "MLAgentBench")
    prompt_file = os.path.join(mla_dir, "prompts", f"prompt_{mode}.txt")
    # read_only_file = os.path.join(source, "read_only_files.txt")
    benchmark_dir = os.path.join(mla_dir, "MLAgentBench", "benchmarks", "task")
    if os.path.exists(benchmark_dir):
        shutil.rmtree(benchmark_dir)
    os.makedirs(benchmark_dir)
    env_dir = os.path.join(benchmark_dir, "env")
    scripts_dir = os.path.join(benchmark_dir, "scripts")
    os.mkdir(scripts_dir)

    # Copy files from source to benchmark
    shutil.copyfile(os.path.join(source, "environment.yml"), os.path.join(mla_dir, "environment.yml"))
    # with open(os.path.join(this_path, "tmp", "environment.yml")) as file:
    #    env_name = file.readline()[6:].strip()
    shutil.copytree(os.path.join(source, "code"), env_dir, symlinks=True)
    shutil.copyfile(os.path.join(source, "experiment.txt"), os.path.join(env_dir, "experiment.txt"))
    shutil.copyfile(os.path.join(source, "paper.txt"), os.path.join(env_dir, "paper.txt"))
    shutil.copyfile(prompt_file, os.path.join(scripts_dir, "research_problem.txt"))

    ### Docker ###
    print("Set up done. Proceeding to create docker image and run...")
    name = f"mla_{mode}_{paper_id}_{exp_id}".lower()
    # Build docker image with given name, build argument (env_name), and path to Dockerfile
    #subprocess.run(["sudo", "docker", "build", "-t", f"{name}", "--build-arg", f"env_name={env_name}", "-f", "./baselines/Dockerfile-mla", "."])
    # subprocess.run(["sudo", "docker", "build", "-t", f"{name}", "-f", "./baselines/Dockerfile-mla-v2", "."])
    # Remove container if exists and run container (with gpu enabled)
    #subprocess.run(["sudo", "docker", "run", "--gpus", "all", "--name", container_name, f"{name}"])
    subprocess.run(["sudo", "docker", "run", "--name", name, "-v", f"{mla_dir}:/app/tmp:ro", "base_image"])
    # After container stops, copy output and log to local
    subprocess.run(["sudo", "docker", "cp", f"{name}:/app/MLAgentBench/MLAgentBench/output.txt", os.path.join(this_path, "tmp")])
    subprocess.run(["sudo", "docker", "cp", f"{name}:/app/MLAgentBench/logs/agent_log/main_log", os.path.join(this_path, "logs", "MLAgentBench", name+"_log")])
    subprocess.run(["sudo", "docker", "rm", name])

    # Retrieve final answer
    answer = 0
    submission_path = os.path.join(this_path, "tmp", "output.txt")
    if os.path.exists(submission_path):
        with open(submission_path, 'r') as sub_file:
            answer = sub_file.readline().strip()
    return answer

def run_baseline(baseline="MLAgentBench", path="../workspace"):
    dirs = path.split(os.sep)
    mode, paper_id, exp_id = dirs[-3], dirs[-2], dirs[-1]
    print(f"Running {baseline} with {mode} on paper {paper_id} experiment {exp_id}")
    if baseline=="MLAgentBench":
        res = run_MLAgentBench(paper_id, exp_id, mode, path)
    else:
        print(f"Baseline {baseline} not supported")
        return 0.0
    try:
        return float(res)
    except Exception as e:
        print("Could not convert to float, using 0.0")
        return 0.0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=str, default="MLAgentBench", help="baseline")
    parser.add_argument("--source", type=str, default="../workspace", help="path to source directory") 

    args = parser.parse_args()
    run_baseline(args.baseline, args.source)


