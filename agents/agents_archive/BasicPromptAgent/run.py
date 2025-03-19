import os
import sys
this_path = os.path.dirname(__file__)
sys.path.append(this_path)

from agents.BasicPromptAgent.agent import CodeAgent

def run_BasicPromptAgent(X, metadata, model, paper, tags, **kwargs):
    agent = CodeAgent(X, metadata, model, paper)
    return agent.run()