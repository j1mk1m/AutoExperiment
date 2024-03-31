import sys 
import json
import os
this_path = os.path.dirname(__file__)
sys.path.append(this_path)

from history import History
from llm import call_llm 

from prompts import base_prompt, tool_prompt

class AutoAgent:
    def __init__(self, env, model, max_steps=50, max_retries=3, **kwargs):
        self.env = env
        self.history = History()
        self.rp_model, self.tc_model, self.mem_model = model, model, model
        self.max_steps = max_steps
        self.max_retries = max_retries
        self.research_plan = "No research plan" # current research plan
        self.memory = "Nothing" # current memory
        self.base_prompt = base_prompt + self.env.get_exp_description()
        
    def run(self):
        for i in range(self.max_steps):
            print(f"Step {i} \n")
            dir_prompt = f"\nCurrent Directory: {self.env.cur_dir} \nDirectory Contents: {self.env.list_files(directory='.')}"

            # 1. Generate Research Plan
            if i == 0:
                rp_prompt = "First, generate a high level research plan, describing the course of action to take. Research plan:"
            else:
                rp_prompt = f"Current research plan: {self.research_plan} \nCurrent Memory: {self.memory}. \n Given this, generate a new research plan, with current status and confirmed results of each step briefly annotated. New research plan and status:"
            
            messages = [{"role": "system", "content": self.base_prompt}, 
                        {"role": "system", "content": dir_prompt},
                        {"role": "user", "content": rp_prompt}]
            new_rp = call_llm(messages, None, self.rp_model).content
            if '1.' in new_rp:
                new_rp = '1.' + new_rp.split('1.')[1]
            self.research_plan = new_rp
            self.history.append_research_plan(new_rp)
            print("Research Plan and Status:")
            print(new_rp)

            # 2a. Action/Tool Call
            tc_prompt = f"Current research plan: {self.research_plan} \nCurrent Memory: {self.memory} \n Given this, select the next function to call."
            messages = [{"role": "system", "content": self.base_prompt}, 
                        {"role": "system", "content": dir_prompt},
                        {"role": "user", "content": tc_prompt}]

            valid = False
            for _ in range(self.max_retries):
                response = call_llm(messages, tool_prompt, self.tc_model)

                if response.tool_calls and len(response.tool_calls) > 0:
                    tool_call = response.tool_calls[0].function
                    action = tool_call.name
                    action_input = json.loads(tool_call.arguments)

                    valid = True
                    break
            
            if not valid:
                return f"No valid response after maximum retries of {self.max_retries}"

            self.history.append_action({"action": action, "arguments": action_input})
            print("\nAgent Response Action/Input:")
            print(action, action_input)

            # 2b. Get observation from envrionment
            observation = self.env.execute(action, action_input)
            if len(observation) > 10000:
                observation = observation[:5000] + "\n...\n" + observation[-5000:]
            self.history.append_observation(observation)
            print("\nObservation:")
            print(observation)

            # 3. Memory Integration 
            mem_prompt = f"Current research plan: {self.research_plan} \nCurrent Memory: {self.memory} \nNew action and observation: Called {action} with input {action_input}. Observation: {observation} \nGenerate content of new memory, keeping any important information from the previous Memory and adding information from the new action and observation."
            messages = [{"role": "system", "content": self.base_prompt}, {"role": "user", "content": mem_prompt}]
            new_mem = call_llm(messages, None, self.mem_model).content
            self.memory = new_mem
            self.history.append_memory(new_mem)
            print("\nMemory:")
            print(new_mem)
            print("\n\n")

            # Exit if final answer is given
            if action == "final_answer":
                self.history.save_history(self.env.mode, self.env.paper_id, self.env.exp_id)
                return action_input.final_answer

        self.history.save_history(self.env.mode, self.env.paper_id, self.env.exp_id)
        return "Maximum steps reached"