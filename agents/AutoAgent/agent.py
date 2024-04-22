import sys
import json
import wandb
import os
this_path = os.path.dirname(__file__)
sys.path.append(this_path)

from history import History
from llm import call_llm 

from prompts import *

class AutoAgent:
    def __init__(self, env, model, tags, max_steps=50, max_retries=3, **kwargs):
        # Configurations
        self.env = env
        self.tags = tags
        self.history = History(tags)
        self.rp_model, self.tc_model, self.mem_model = model, model, model
        self.max_steps = max_steps
        self.max_retries = max_retries
        self.research_plan = "None"
        self.mode = self.env.mode
        self.v = True

        # Prompts
        self.experiment = self.env.get_exp_description()
        if "FC" in self.mode:
            self.base_prompt = base_prompt_FC.format(experiment=self.experiment)
            self.tc_prompt = tc_prompt_FC.format(experiment=self.experiment)
        else:
            self.base_prompt = base_prompt_PC.format(experiment=self.experiment, func_name=self.env.X["func_to_block"]["name"], file_name=self.env.X["func_to_block"]["script"])
            self.tc_prompt = tc_prompt_PC.format(experiment=self.experiment, func_name=self.env.X["func_to_block"]["name"], file_name=self.env.X["func_to_block"]["script"])
        self.tools = tool_prompt 
        if "PC" in self.mode:
            self.tools += pc_tool
        if "refsol" in self.mode:
            self.tools += refsol_tool
        if self.v: print("Base prompt", self.base_prompt)
        if self.v: print("Available tools", [tool["function"]["name"] for tool in self.tools])
            
    def run(self):
        messages = [{"role": "system", "content": self.base_prompt}]
        for i in range(self.max_steps):
            if self.v: print(f"Step {i}")
            wandb.log({"step": i})
            
            # 1. Generate Research Plan
            prompt = rp_prompt.format(plan=self.research_plan) if "refsol" not in self.mode else rp_prompt_refsol.format(plan=self.research_plan) 
            messages.append({"role": "system", "content": prompt}) 
            new_rp = call_llm(messages, None, self.rp_model).content
            self.research_plan = new_rp
            messages.append({"role": "assistant", "content": new_rp})

            if self.v: print(f"Research Plan: \n{new_rp}\n")
            self.history.append_research_plan(new_rp)

            # 2a. Action/Tool Call 
            messages.append({"role": "system", "content": self.tc_prompt})
            valid = False
            for _ in range(self.max_retries):
                response = call_llm(messages, self.tools, self.tc_model)

                if response and response.tool_calls and len(response.tool_calls) > 0:
                    valid = True
                    messages = messages[:-1]
                    messages.append(response)
                    break

                if response is None:
                    # most likely token limit: remove history except the current research plan
                    messages = messages[0:1] + messages[-2:]
                 
            if not valid:
                self.history.save_history()
                return f"No valid response after maximum retries."

            # 2b. Get observation from envrionment
            for tool_call in response.tool_calls:
                action = tool_call.function.name
                action_input = json.loads(tool_call.function.arguments)
                if self.v: print(f"Action: {action} Inputs: {action_input}")
                observation = self.env.execute(action, action_input)
                if len(observation) > 10000:
                    # TODO better summarization of long observations
                    observation = observation[:5000] + "\n...\n" + observation[-5000:]
                messages.append({"role": "tool", "tool_call_id": tool_call.id, "name": action, "content": observation})
                self.history.append_action(str(tool_call.function))
                self.history.append_observation(observation)
                if self.v: print(f"Observation: {observation}\n")

            # Exit if final answer is given
            if action == "final_answer":
                self.history.save_history()
                return action_input["final_answer"]

        self.history.save_history()
        return "Maximum steps reached"
