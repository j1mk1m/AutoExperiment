import os
import argparse
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from MLAgentBench import LLM
from MLAgentBench.environment import Environment
from MLAgentBench.agents.agent import Agent, SimpleActionAgent, ReasoningActionAgent
from MLAgentBench.agents.agent_research import ResearchAgent
from MLAgentBench.schema import Action

this_dir = os.path.dirname(__file__)

def run(agent_cls, args):
    with Environment(args) as env:

        print("=====================================")
        research_problem, benchmark_folder_name = env.get_task_description()
        print("Benchmark folder name: ", benchmark_folder_name)
        print("Research problem: ", research_problem)
        print("Lower level actions enabled: ", [action.name for action in env.low_level_actions])
        print("High level actions enabled: ", [action.name for action in env.high_level_actions])
        print("Read only files: ", env.read_only_files, file=sys.stderr)
        print("=====================================")  
        
        env.execute(Action("Change Directory", {"dir_name": "code"}))

        agent = agent_cls(args, env)
        final_message = agent.run(env)
        print("=====================================")
        print("Final message: ", final_message)

def main(combined_id):
    with open(os.path.join(this_dir, "config.yml"), "r") as yml_file:
        args = yaml.safe_load(yml_file)["parameters"]

    args = argparse.Namespace(**args)
    args.log_dir = os.path.join(this_dir, args.log_dir, combined_id)
    args.work_dir = os.path.join(this_dir, args.work_dir, combined_id)
    args.task = combined_id
    print(args, file=sys.stderr)
    if args.no_retrieval or args.agent_type != "ResearchAgent":
        # should not use these actions when there is no retrieval
        args.actions_remove_from_prompt.extend(["Retrieval from Research Log", "Append Summary to Research Log", "Reflection"])
    LLM.FAST_MODEL = args.fast_llm_name
    run(getattr(sys.modules[__name__], args.agent_type), args)
    
if __name__=="__main__":
    combined_id = sys.argv[1]
    main(combined_id)

