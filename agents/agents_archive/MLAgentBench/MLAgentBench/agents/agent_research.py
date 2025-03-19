""" This file contains the agent class for our AI research agent."""
import os
import json
import sys
import wandb
import anthropic
import tiktoken
import json
from MLAgentBench.LLM import complete_text_fast, complete_text, function_calling_openai
from MLAgentBench.schema import Action
from .agent import Agent

enc = tiktoken.get_encoding("cl100k_base")

initial_prompt = """You are a helpful research assistant. You have access to the following tools:
{tools_prompt}

Research Problem: {task_description}

"""

response_prompt = """Always respond in this format exactly:
{format_prompt}
"""

function_call_prompt = """You are a helpful research assistant. 

Research Problem: {task_description}

Follow these instructions and do not forget them:
- First, come up with a high level plan based on your understanding of the problem and available tools and record it in the Research Plan and Status. You can revise the plan later.
- Research Plan and Status should well organized and succinctly keep track of 1) high level plan (can be revised), 2) what steps have been done and what steps are in progress, 3) short results and conclusions of each step after it has been performed. 
- Performance numbers and estimates can only be confirmed and included in the status by running the code and observing the output.
- In your response, always pick exactly one tool call. The tools allow you to view, modify, and execute files in the environment.
- If you believe you have solved the problem, you can use the Final Answer action to submit your answer. You can only submit once, so double check that you have achieved the goal before submitting.
"""

format_prompt_dict = {
    "Reflection": "What does the observation mean? If there is an error, what caused the error and how to debug?",
    "Research Plan and Status": "The full high level research plan, with current status and confirmed results of each step briefly annotated. It must only include progress that has been made by previous steps. If there is any update, enclose the new update text in double asterisks **like this**. If there is no update, just copy the previous step Research Plan and Status. The high level plan from the previous step should be fully retained, unless it is intentionally revised.",
    "Fact Check": "List all objective statements in the updates to Research Plan and Status one by one and point out whether it is guessed versus directly confirmed by the previous observation directly above. Performance numbers can only be confirmed by running the code and observing the output.",
    "Thought": "What you are currently doing, what actions to perform and why",
    "Action": "the action to take, should be one of the names of the tools",
    "Action Input": "the input to the action as a valid JSON string",
}

tools = [
    {
        "type": "function",
        "function": {
            "name": "understand_file",
            "description": "Use this to read the whole file and understand certain aspects. You should provide detailed description on what to look for and what should be returned. To get a better understanding of the file, you can use Inspect Script Lines action to inspect specific part of the file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_name": {
                        "type": "string",
                        "description": "a valid file name with relative path to current directory if needed",
                    },
                    "things_to_look_for": {
                        "tupe": "string",
                        "description": "a detailed description on what to look for and what should returned"
                    }
                }
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "inspect_script_lines",
            "description": "Use this to inspect specific part of a python script precisely, or the full content of a short script. The number of lines to display is limited to 100 lines. This is especially helpful when debugging.",
            "parameters": {
                "type": "object",
                "properties": {
                    "script_name": {
                        "type": "string",
                        "description": "a valid python script name with relative path to current directory if needed"
                    },
                    "start_line_number": {
                        "type": "number",
                        "description": "a valid line number"
                    },
                    "end_line_number": {
                        "type": "number",
                        "description": "a valid line number"
                    }
                }
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "edit_script",
            "description": "Use this to do a relatively large but cohesive edit over a python script. Instead of editing the script directly, you should describe the edit instruction so that another AI can help you do this.",
            "parameters": {
                "type": "object",
                "properties": {
                    "script_name": {
                        "type": "string",
                        "description": "a valid python script name with relative path to current directory if needed. An empty sctipt will be created if it does not exist."
                    },
                    "edit_instruction": {
                        "type": "string",
                        "description": "a detailed step by step description on how to edit it."
                    },
                    "save_name": {
                        "type": "string",
                        "description": "a valid file name with relative path to current directory if needed"
                    }
                }
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "Use this to list the files and directories",
            "parameters": {
                "type": "object",
                "properties": {
                    "dir_path": {
                        "type": "string",
                        "description": "a valid relative path to a directory, such as \".\" or \"folder1/folder2\""
                    }
                },
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "execute_script",
            "description": "Use this to execute the python script. The script must already exist.",
            "parameters": {
                "type": "object",
                "properties": {
                    "script_name": {
                        "type": "string",
                        "description": "a valid python script name with relative path to current directory if needed and arguments appended as a single string if needed"
                    }
                }
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "execute_bash",
            "description": "Use this to execute a bash script. The script must already exist",
            "parameters": {
                "type": "object",
                "properties": {
                    "script_name": {
                        "type": "string",
                        "description": "a valid bash script with relative path to current directory and arguments appended as a single string if needed"
                    }
                }
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "final_answer",
            "description": "Use this to submit the final answer to the current task",
            "parameters": {
                "type": "object",
                "properties": {
                    "final_answer": {
                        "type": "number",
                        "description": "numeric value of the final experiment result"
                    }
                },
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "change_directory",
            "description": "Use this to navigate the file structure",
            "parameters": {
                "type": "object",
                "properties": {
                    "dir_path": {
                        "type": "string",
                        "description": "valid path to directory"
                    }
                }
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "command_line",
            "description": "Use this to run any linux command line command",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "valid linux command line command"
                    }
                }
            }
        }
    },
]

func_to_name_map = {
    "understand_file": "Understand File",
    "inspect_script_lines": "Inspect Script Lines",
    "edit_script": "Edit Script",
    "change_directory": "Change Directory",
    "final_answer": "Final Answer",
    "execute_bash": "Execute Shell Script",
    "execute_script": "Execute Script",
    "list_files": "List Files",
    "command_line": "Command Line",
}   

class ResearchAgent(Agent):
    """This class implements AI research agent with different configurations."""

    def __init__(self, args, env):
        super().__init__(args, env)
        self.func_call = False and "gpt" in args.llm_name # FALSE if OpenAI model, use function calling API
        self.valid_format_entries = ["Reflection",  "Research Plan and Status","Fact Check", "Thought"] # use all entries by default
        self.action_entries = ["Action", "Action Input"]
        if self.func_call:
            self.initial_prompt  = function_call_prompt.format(task_description=env.research_problem)
        else:
            self.valid_format_entries += self.action_entries
            self.initial_prompt = initial_prompt.format(tools_prompt=self.tools_prompt, tool_names=self.prompt_tool_names,  task_description=env.research_problem)
        self.response_prompt = response_prompt.format(format_prompt="\n".join([f"{k}: {format_prompt_dict[k]}" for k in self.valid_format_entries]))

    def run(self, env):
        last_steps = self.args.max_steps_in_context
        last_observation_step = self.args.max_observation_steps_in_context

        with open(os.path.join(self.log_dir , "main_log"), "a", 1) as f:
            f.write(self.initial_prompt + "\n")

        while not env.is_final() and len(self.history_steps) < self.args.agent_max_steps:

            curr_step = len(self.history_steps)
            print(f"Step {curr_step}")

            #### call LLM for next action ###

            ###########################################################
            #     construct prompt for LLM based on previous steps    #
            ###########################################################

            prompt = self.initial_prompt
            messages = [{"role" : "system", "content" : self.initial_prompt }]
            if curr_step > 0:
                if not self.args.no_retrieval:

                    # retrieval action
                    relevant_history = env.execute(Action("Retrieval from Research Log", {"current_plan": ""}))

                    prompt += f"""
        Here is a summary of relevant actions and observations you have done:
        ```
        {relevant_history}
        ```
        Here are the exact several steps you have done most recently (up to 3 steps):
        """
                    messages.append({"role" : "system", "content" : f"Here is a summary of relevant actions and observations you have done {relevant_history}"})
            else:
                prompt += "\nNow let's start!\n\n"

            for idx in range(max(curr_step - last_steps, 0), curr_step):
                action_string = ""
                action_string = self.print_action(self.history_steps[idx]["action"], self.valid_format_entries)

                history = anthropic.AI_PROMPT + "\n"+ action_string + "\nObservation:"
                prompt += anthropic.AI_PROMPT + "\n"+ action_string + "\nObservation:"
                if curr_step - idx > last_observation_step:
                    prompt += "<Done>\n\n"
                else:
                    try:
                        prompt += "\n```\n" + self.history_steps[idx]["observation"] + "\n```\n\n"
                        messages.append({"role" : "system", "content" : "Here is a past action you have done and its corresponding output:\n"+ action_string + "\nObservation:" + "\n```\n" + self.history_steps[idx]["observation"] + "\n```\n\n"})
                    except:
                        import pdb; pdb.set_trace()

            if self.func_call:
                messages.append({"role": "system", "content": self.response_prompt})
            else:
                prompt += self.response_prompt
                
            ###############################################
            #     call LLM until the response is valid    #
            ###############################################

            entries = None
            valid_response = False
            for _ in range(self.args.max_retries):
                log_file = os.path.join(self.log_dir , f"step_{curr_step}_log.log")
                #print("Prompting ResearchAgent: ", prompt)
                if self.func_call:
                   # print(f"Prompting LLM: {messages}")
                    response = function_calling_openai(messages, tools, self.args.llm_name)
                    try:
                        response = response.choices[0].message

                        if response.tool_calls and len(response.tool_calls) > 0:
                            # Check that tool is called
                            tool_call = response.tool_calls[0].function

                            content = response.content if response.content else ""
                            entries = self.parse_entries(content, self.valid_format_entries)
                            entries["Action"] = func_to_name_map[tool_call.name]
                            entries["Action Input"] = json.loads(tool_call.arguments)
                            valid_response = True
                    except Exception as e:
                        print(f"Error processing LLM response {e}", file=sys.stderr)
                        messages.append({"role": "system", "content": "Your response was in incorrect format. Please provide a valid response with a tool call and all of the following entries in text as content: " + ", ".join(self.valid_format_entries)})
                    else:
                        break
                else:
                    completion = complete_text(prompt, log_file, self.args.llm_name)

                    try:
                        entries = self.parse_entries(completion, self.valid_format_entries)
                        assert entries["Action"].strip() in self.all_tool_names
                        #assert "Action Input" in entries.keys()
                        valid_response = True
                    except:
                        print("Response is invalid and discarded", file=sys.stderr)
                        prompt += "\n\n Your response was in incorrect format. Please provide a valid response with all entries: " + ", ".join(self.valid_format_entries) + "\n\n"
                    else:
                        break
            if not valid_response:
                return "No valid response after max_retries"

            ########################################################
            #     postprocess LLM output and parse to env actions  #
            ########################################################

            for e in self.valid_format_entries:
                if e not in entries:
                    entries[e] = ""

            rg = entries["Research Plan and Status"]
            action = entries["Action"].strip()
            if not self.func_call:
                raw_action_input = entries["Action Input"]
            else:
                action_input = entries["Action Input"]
                entries["Action Input"] = json.dumps(entries["Action Input"])

            new_research_plan_content = rg.strip("```") + "\n\n" 
            entries["Research Plan and Status"] = new_research_plan_content
            entries["Research Plan and Status"] = new_research_plan_content.replace("**", "")
 
            # parse the action input if we can ; other wise just return the original input and wait env to throw back an error
            if not self.func_call:
                parsing_error = ""
                try:
                    action_input = self.parse_action_input(raw_action_input, self.action_infos[action])
                except Exception as e:
                    action_input = raw_action_input
                    parsing_error = str(e)
                
            
            with open(os.path.join(self.log_dir , "main_log"), "a", 1) as f:
                f.write("Step " + str(curr_step) + ":\n")
                f.write(anthropic.AI_PROMPT + "\n" + self.print_action(entries, self.valid_format_entries) + "\nObservation:\n")


            ########################################
            #         execute action in env        #
            ########################################

            if action == "Final Answer":
                return action_input["final_answer"]

            if type(action_input) == dict:
                observation = env.execute(Action(action, action_input))
            else:
                # parsing failed, give agent parsing error
                usage = ",\n            ".join([f"{k}: [{v}]" for k, v in self.action_infos[action].usage.items()])
                usage = f"""{{
            {usage}
}}"""
                invalid_action_error = f"The action input for {action} needs to be a valid json with proper entries. You may have missed the comma between entries or used triple quotes (json does not recognizes triple quotes). Please use the correct format and try again:\n{usage}"

                observation = "ActionInputParsingError: "+ parsing_error + "\n" + invalid_action_error


            #######################################################
            #               update history_steps                  #
            #######################################################

            # if observation is too long, we need to summarize it
            if len(observation) > 5000:
                log_file = os.path.join(self.log_dir , f"step_{curr_step}_summarize_observation_log.log")

                print("Observation is too long. Summarizing...", file=sys.stderr)
                if action == "Execute Script" or action == "Execute Bash Script":
                    observation = "Output of script is too long, the last outputs are: " + observation[-5000:]
                else:
                    observation = self.summarize_observation(self.print_action(entries, self.valid_format_entries), observation, log_file)

            self.history_steps.append({"step_idx": len(env.trace.steps), "action": entries, "observation": observation})

            # wandb logging
            # wandb.log({"step": curr_step, "prompt": prompt, "response": entries, "observation": observation})

            ## filter out ActionInputParsingError if last step is not action input parsing error
            if not observation.startswith("ActionInputParsingError"):
                self.history_steps = [step for step in self.history_steps if not step["observation"].startswith("ActionInputParsingError")]

            with open(os.path.join(self.log_dir , "main_log"), "a", 1) as f:
                f.write("\n```\n" + self.history_steps[-1]["observation"] + "\n```\n\n")


            #######################################################
            #      write to research log for retrieval            #
            #######################################################
            if not self.args.no_retrieval:
                summary_of_last_step = "Too long to summarize."
                for _ in range(self.args.max_retries):
                    try:
                        log_file = os.path.join(self.log_dir , f"step_{curr_step}_summary_log.log")
                        summary_of_last_step = self.summarize_action_and_observation(self.print_action(self.history_steps[-1]["action"], self.valid_format_entries), self.history_steps[-1]["observation"], log_file = log_file)
                        break
                    except Exception as e:
                        print(e)
                        print("Trying again.")

                action = "Append Summary to Research Log"
                action_input = { "content": "\n\nStep " + str(curr_step) + ":\n" + summary_of_last_step + "\n"}
                env.execute(Action(action, action_input))

            step_idx = len(env.trace.steps) - 1
            # self.save(os.path.join(self.log_dir , f"agent_{step_idx}_{curr_step}.json"))

        if env.is_final():
            return "Finished due to env.is_final() == True"
        else:
            return "Finished due to agent max steps reached"


    ################### Helper functions #####################

    def summarize_observation(self, action, observation, log_file, bs = 10000):
        """ Summarize the observation if it is too long with a sliding window of size bs """

        bs = 10000
        blocks = [observation[i:i+bs] for i in range(0, len(observation), bs)]
        descriptions = []
        for idx, b in enumerate(blocks):
            start_line_number = bs*idx+1
            end_line_number = bs*idx+1 + len(b)
            prompt = f"""
{action}

The full observation is too long. Given this (partial) observation from character {start_line_number} to character {end_line_number}: 
``` 
{b}
```
Summarize the observation concisely in this format:
[Observation]: Summarize all relevant details in the observation objectively

Do not include any result that is guessed rather than directly confirmed by the observation. Do not include additional information or suggestions.
"""

            completion = complete_text_fast(prompt, log_file=log_file +f"_{idx}")
            descriptions.append(completion)
        if len(descriptions) == 1:
            completion = descriptions[0]
        else:
            descriptions = "\n\n".join(["Segment {idx}: \n\n" + s for s in descriptions])[-bs:]
            prompt = f"""
{action}

The full observation is too long. 
Given summaries for each segments of the whole observation, summarize to get a cohesive description of the entire observation.
{descriptions}

Summarize the observation concisely in this format:
[Observation]: Summarize all relevant details in the observation objectively

Do not include any result that is guessed rather than directly confirmed by the observation. Do not include additional information or suggestions.
"""

            completion = complete_text_fast(prompt, log_file=log_file)
        try:
            return completion.split("[Observation]:")[1]
        except:
            return completion

    @staticmethod
    def summarize_action_and_observation(action, observation, **kwargs):
        """ Summarize the action and observation to an entry in the research log """

        prompt = f"""Given your action and the observation: 
        {action} 
        [Observation]:
        ```
        {observation}
        ```
        Summarize your action and the observation in this format:
        [Reasoning]: Summarize the reasoning behind the action
        [Action]: Summarize all relevant details of the action objectively
        [Observation]: Summarize all relevant details in the observation objectively
        Do not include any result that is guessed rather than directly confirmed by the observation. Do not include additional information or suggestions.
        """

        summary = "[Reasoning]:" + complete_text_fast(prompt, log_file=kwargs["log_file"]).split("[Reasoning]:")[1]
        return summary
    
