import sys 
import os
import wandb
import shutil
this_path = os.path.dirname(__file__)
sys.path.append(this_path)

from agent import AutoAgent
from environment import Environment

def run_AutoAgent(X, metadata, model, retrieval, tags, **kwargs):
    env = Environment(X, metadata, model, retrieval, **kwargs)
    agent = AutoAgent(env, model, tags, retrieval=retrieval, **kwargs)
    res = agent.run()
    wandb.log({"final_answer": res})
    shutil.rmtree(env.workspace_root)
    return res
