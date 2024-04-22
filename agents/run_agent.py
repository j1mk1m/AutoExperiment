import argparse
import os
import sys
this_path = os.path.dirname(__file__)
sys.path.append(".")

# Agents
from dataset.run_refsol import run_refsol
from agents.MLAgentBench.run import run_MLAgentBench
from agents.AutoAgent.run import run_AutoAgent
from agents.BasicPromptAgent.run import run_BasicPromptAgent

def parse_output(res):
    # parse result to extract numerical value
    try:
        if isinstance(res, int): return float(res)
        if isinstance(res, float): return res
        res = str(res)
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

def run_agent(agent="MLAgentBench", X=None, model="gpt-3.5-turbo-1106", tags="", **kwargs):
    if agent=="MLAgentBench":
        res = run_MLAgentBench(X, model, **kwargs)
    elif agent=="refsol":
        res = run_refsol(X)
    elif agent=="AutoAgent":
        res = run_AutoAgent(X, model, tags, **kwargs)
    elif agent=="BasicPromptAgent":
        res = run_BasicPromptAgent(X, model, tags, **kwargs)
    else:
        print(f"Agent {agent} not supported, returning 0.0")
        return 0.0

    return parse_output(res)
