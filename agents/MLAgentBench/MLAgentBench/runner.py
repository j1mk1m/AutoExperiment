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
from MLAgentBench.agents.agent_langchain  import LangChainAgent
try:
    from MLAgentBench.agents.agent_autogpt  import AutoGPTAgent
except:
    print("Failed to import AutoGPTAgent; Make sure you have installed the autogpt dependencies if you want to use it.")

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
        with open(os.path.join(this_dir, "output.txt"), 'w') as output_file:
            output_file.write(final_message)
        print("=====================================")
        print("Final message: ", final_message)
        wandb.log({"final": final_message})

    env.save("final")


def main(combined_id):
    with open(os.path.join(this_dir, "config.yml"), "r") as yml_file:
        args = yaml.safe_load(yml_file)["parameters"]
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="debug", help="task name")
    parser.add_argument("--log-dir", type=str, default=os.path.join(this_dir, "../", "logs"), help="log dir")
    parser.add_argument("--work-dir", type=str, default=os.path.join(this_dir, "../", "workspace"), help="work dir")
    parser.add_argument("--device", type=int, default=0, help="device id")
    parser.add_argument("--python", type=str, default="python", help="python command")
    parser.add_argument("--interactive", action="store_true", help="interactive mode")
    parser.add_argument("--resume", type=str, default=None, help="resume from a previous run")
    parser.add_argument("--resume-step", type=int, default=0, help="the step to resume from")

    # general agent configs
    parser.add_argument("--agent-type", type=str, default="ResearchAgent", help="agent type")
    parser.add_argument("--llm-name", type=str, default="claude-v1", help="llm name") # edit
    parser.add_argument("--fast-llm-name", type=str, default="claude-v1", help="llm name") # edit
    parser.add_argument("--edit-script-llm-name", type=str, default="claude-v1", help="llm name") # edit
    parser.add_argument("--edit-script-llm-max-tokens", type=int, default=4000, help="llm max tokens") # edit
    parser.add_argument("--agent-max-steps", type=int, default=50, help="max iterations for agent") # edit
    parser.add_argument("--max-steps", type=int, default=50, help="number of steps") # edit
    parser.add_argument("--max-time", type=int, default=5* 60 * 60, help="max time") # edit

    # research agent configs
    parser.add_argument("--actions-remove-from-prompt", type=str, nargs='+', default=[], help="actions to remove in addition to the default ones: Read File, Write File, Append File, Retrieval from Research Log, Append Summary to Research Log, Python REPL, Edit Script Segment (AI), Undo Edit Script, Copy File")
    parser.add_argument("--actions-add-to-prompt", type=str, nargs='+', default=[], help="actions to add")
    parser.add_argument("--no-retrieval", action="store_true", help="disable retrieval")
    parser.add_argument("--max-steps-in-context", type=int, default=3, help="max steps in context") # edit
    parser.add_argument("--max-observation-steps-in-context", type=int, default=3, help="max observation steps in context") # edit
    parser.add_argument("--max-retries", type=int, default=5, help="max retries")

    # langchain configs
    parser.add_argument("--langchain-agent", type=str, default="zero-shot-react-description", help="langchain agent")

    args = parser.parse_args()
    """

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
