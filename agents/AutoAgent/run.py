import sys 
import os
import wandb
this_path = os.path.dirname(__file__)
sys.path.append(this_path)

from agent import AutoAgent
from environment import Environment

def run_AutoAgent(X, model, tags, **kwargs):
    env = Environment(X, model, **kwargs)
    print(env.edit_missing_function(instruction="Fill out this missing function"))
    return 0
    # agent = AutoAgent(env, model, tags, **kwargs)
    # res = agent.run()
    # wandb.log({"final_answer": res})
    # return res
