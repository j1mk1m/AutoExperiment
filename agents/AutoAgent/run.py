import sys 
import os
import wandb
this_path = os.path.dirname(__file__)
sys.path.append(this_path)

from agent import AutoAgent
from environment import Environment

def run_AutoAgent(paper_id, exp_id, mode, source, model, tags, **kwargs):
    env = Environment(paper_id, exp_id, mode, source, model, **kwargs)
    agent = AutoAgent(env, model, tags, **kwargs)
    res = agent.run()
    wandb.log({"final_answer": res})
    return res
