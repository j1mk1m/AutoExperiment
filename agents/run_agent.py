import argparse
import wandb
import os
import sys

this_path = os.path.dirname(__file__)
sys.path.append(".")
from dataset.run_refsol import run_refsol
from agents.MLAgentBench.run import run_MLAgentBench
from agents.AutoAgent.run import run_AutoAgent

openai_api_key = open(os.path.join(this_path, "openai_api_key.txt")).read().strip()
# os.environ["OPENAI_API_KEY"] = openai_api_key

def run_agent(agent="MLAgentBench", path="../workspace", verbose=False, model="gpt-3.5-turbo-1106", **kwargs):
    dirs = path.split(os.sep)
    mode, paper_id, exp_id = dirs[-3], dirs[-2], dirs[-1]
    if verbose:
        print(f"Running {agent} with {mode} on paper {paper_id} experiment {exp_id}")

    if agent=="MLAgentBench":
        res = run_MLAgentBench(paper_id, exp_id, mode, path, model, **kwargs)
    elif agent=="refsol":
        res = run_refsol(paper_id, exp_id, mode, path)
    elif agent=="AutoAgent":
        res = run_AutoAgent(paper_id, exp_id, mode, path, model, **kwargs)
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
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    
    run_agent(args.agent, args.source, args.verbose)


