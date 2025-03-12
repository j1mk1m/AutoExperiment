""" 
This file is the entry point for MLAgentBench.
"""
import os
import argparse
import sys
import yaml
import wandb
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from MLAgentBench import LLM
from MLAgentBench.environment import Environment
from MLAgentBench.agents.agent import Agent, SimpleActionAgent, ReasoningActionAgent
from MLAgentBench.agents.agent_research import ResearchAgent

this_dir = os.path.dirname(__file__)

def run(agent_cls, args):
    with Environment(args) as env:

        print("=====================================")
        research_problem, benchmark_folder_name = env.get_task_description()
        print("Benchmark folder name: ", benchmark_folder_name)
        print("Research problem: ", research_problem)

        agent = agent_cls(args, env)

        print("Actions enabled: ", agent.prompt_tool_names)
        print("=====================================")  
        try:
            final_message = agent.run(env)
        except Exception as e:
            final_message = f"Got error during agent run: {e}"

        #with open(os.path.join(this_dir, "output.txt"), 'w') as output_file:
        #    output_file.write(final_message)
        print("=====================================")
        print("Final message: ", final_message)
        wandb.log({"final": final_message})
        return final_message


def main(X, metadata, model):
    combined_id = metadata["combined_id"]
    with open(os.path.join(this_dir, "config.yml"), "r") as yml_file:
        args = yaml.safe_load(yml_file)["parameters"]

    args = argparse.Namespace(**args)
    args.log_dir = os.path.join(this_dir, args.log_dir, model, combined_id)
    args.work_dir = os.path.join(this_dir, args.work_dir)
    args.task = combined_id
    args.no_retrieval = False
    LLM.FAST_MODEL = args.fast_llm_name
    args.llm_name = model
    print(args, file=sys.stderr)
    return run(getattr(sys.modules[__name__], args.agent_type), args)
    