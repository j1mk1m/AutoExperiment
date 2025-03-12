import argparse
import os
import sys
import json
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
        return json.loads(res)
    except Exception as e:
        print(f"Could not convert {res} to dict")
        return res

def run_agent(agent="AutoAgent", X=None, metadata=None, model="gpt-4o-mini", retrieval="agent", tags="", **kwargs):
    if agent=="MLAgentBench":
        res = run_MLAgentBench(X, metadata, model, **kwargs)
    elif agent=="refsol":
        res = run_refsol(X, metadata)
    elif agent=="AutoAgent":
        res = run_AutoAgent(X, metadata, model, retrieval=retrieval, tags=tags, **kwargs)
    elif agent=="BasicPromptAgent":
        res = run_BasicPromptAgent(X, metadata, model, retrieval, tags, **kwargs)
    else:
        print(f"Agent {agent} not supported, returning 0.0")
        return 0.0

    return parse_output(res)
