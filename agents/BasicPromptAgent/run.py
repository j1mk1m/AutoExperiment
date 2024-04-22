import os
import sys
this_path = os.path.dirname(__file__)
sys.path.append(this_path)

from agents.BasicPromptAgent.agent import CodeAgent

def run_BasicPromptAgent(X, model, tags, **kwargs):
    agent = CodeAgent(X, model)
    return agent.run()