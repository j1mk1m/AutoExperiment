import sys 
import json
import os
this_path = os.path.dirname(__file__)
sys.path.append(this_path)

from history import History
from llm import call_llm 

from prompts import base_prompt, tool_prompt, rp_prompt, tc_prompt, mem_prompt, v2_rp_prompt

class AutoAgent:
    def __init__(self, env, model, tags, max_steps=50, max_retries=3, **kwargs):
        self.env = env
        self.tags = tags
        self.history = History(tags)
        self.rp_model, self.tc_model, self.mem_model = model, model, model
        self.max_steps = max_steps
        self.max_retries = max_retries
        self.research_plan = "No research plan" # current research plan
        self.memory = "Nothing" # current memory
        self.base_prompt = base_prompt + self.env.get_exp_description()
        self.v = True
        
    def run(self):
        for i in range(self.max_steps):
            if self.v: print(f"Step {i} \n")
            dir_prompt = f"The current directory is {self.env.cur_dir} \nDirectory Contents: {self.env.list_files(directory='.')}"

            # 1. Generate Research Plan
            if i == 0:
                prompt = "First, generate a high level research plan, describing the course of action to take."
            else:
                prompt = rp_prompt.format(research_plan=self.research_plan, memory=self.memory)
            
            messages = [{"role": "system", "content": self.base_prompt}, 
                        {"role": "system", "content": dir_prompt},
                        {"role": "user", "content": prompt}]
            new_rp = call_llm(messages, None, self.rp_model).content
            if '1.' in new_rp:
                new_rp = '1.' + new_rp.split('1.')[1]
            self.research_plan = new_rp
            self.history.append_research_plan(new_rp)
            if self.v: print("Research Plan and Status:")
            if self.v: print(new_rp)

            # 2a. Action/Tool Call
            prompt = tc_prompt.format(research_plan=self.research_plan, memory=self.memory)
            messages = [{"role": "system", "content": self.base_prompt}, 
                        {"role": "system", "content": dir_prompt},
                        {"role": "user", "content": prompt}]

            valid = False
            for _ in range(self.max_retries):
                response = call_llm(messages, tool_prompt, self.tc_model)

                if response.tool_calls and len(response.tool_calls) > 0:
                    tool_call = response.tool_calls[0].function
                    action = tool_call.name
                    action_input = json.loads(tool_call.arguments)
                    reasoning = response.content

                    valid = True
                    break
                
                messages.append({"role": "user", "content": "Make sure to choose a tool call."})
            
            if not valid:
                return f"No valid response after maximum retries of {self.max_retries}"

            self.history.append_action({"action": action, "arguments": action_input, "reasoning": reasoning})
            if self.v: print("\nAgent Response Action/Input:")
            if self.v: print(action, action_input, f"Reasoning: {reasoning}")

            # 2b. Get observation from envrionment
            observation = self.env.execute(action, action_input)
            if len(observation) > 10000:
                # TODO better summarization of long observations
                observation = observation[:5000] + "\n...\n" + observation[-5000:]
            self.history.append_observation(observation)
            if self.v: print("\nObservation:")
            if self.v: print(observation)

            # 3. Memory Integration 
            prompt = mem_prompt.format(research_plan=self.research_plan, memory=self.memory, action=action, action_input=action_input, observation=observation)
            messages = [{"role": "system", "content": self.base_prompt}, 
                        {"role": "user", "content": prompt}]
            new_mem = call_llm(messages, None, self.mem_model).content
            self.memory = new_mem
            self.history.append_memory(new_mem)
            if self.v: print("\nMemory:")
            if self.v: print(new_mem)
            if self.v: print("\n\n")

            # Exit if final answer is given
            if action == "final_answer":
                self.history.save_history(self.env.mode, self.env.paper_id, self.env.exp_id)
                return action_input["final_answer"]

        self.history.save_history(self.env.mode, self.env.paper_id, self.env.exp_id)
        return "Maximum steps reached"

    def run_v2(self):
        messages = [{"role": "system", "content": self.base_prompt}]
        for i in range(self.max_steps):
            if self.v: print(f"Step {i}")
            # 1. Generate Research Plan
            if i == 0:
                prompt = "First, generate a high level research plan, describing the course of action to take."
            else:
                prompt = v2_rp_prompt

            messages.append({"role": "system", "content": prompt})
            
            new_rp = call_llm(messages, None, self.rp_model).content
            self.research_plan = new_rp
            messages.append({"role": "assistant", "content": new_rp})
            if self.v: print(f"Research Plan: \n{new_rp}")

            # 2a. Action/Tool Call 
            valid = False
            for _ in range(self.max_retries):
                response = call_llm(messages, tool_prompt, self.tc_model)

                if response.tool_calls and len(response.tool_calls) > 0:
                    valid = True
                    messages.append(response)
                    break
                
                messages.append({"role": "system", "content": "Make sure to choose a tool call."})
            
            if not valid:
                return f"No valid response after maximum retries."

            # 2b. Get observation from envrionment
            for tool_call in response.tool_calls:
                action = tool_call.function.name
                action_input = json.loads(tool_call.function.arguments)
                observation = self.env.execute(action, action_input)
                if len(observation) > 10000:
                    # TODO better summarization of long observations
                    observation = observation[:5000] + "\n...\n" + observation[-5000:]
                messages.append({"role": "tool", "tool_call_id": tool_call.id, "name": action, "content": observation})
                if self.v: print(f"Action {action} Inputs {action_input} \nObservation: {observation}")

            # Exit if final answer is given
            if action == "final_answer":
                return action_input["final_answer"]

        return "Maximum steps reached"
