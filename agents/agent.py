import json
import prompts
from abc import ABC, abstractmethod

def add_agent_args(parser):
    parser.add_argument("--agent", type=str, choices=["refsol", "ReAct", "Planning", "MLAgentBench", "Reflexion"], required=True,
                        help="Type of agent to use")
    parser.add_argument("--max_retries", type=int, default=3,
                        help="Maximum number of retries for LLM calls")


class Agent(ABC):
    def __init__(self, env, llm_manager, memory, X, metadata, max_retries=3) -> None:
        self.env = env
        self.tools = self.env.get_tool_info()
        self.llm_manager = llm_manager
        self.memory = memory

        # Datapoint
        self.X = X
        self.metadata = metadata

        # agent configs
        self.max_retries = max_retries

        # Prompts
        self.tool_descriptions = self.env.get_tool_descriptions()
        experiment = self.env.get_exp_description()
        self.system_prompt = prompts.system_prompt.format(experiment=experiment, tools=self.tool_descriptions)
        self.thought_prompt = prompts.react_prompt 
        self.thought_reprompt = prompts.react_reprompt 

    def step(self, last_step):
        prompt = self._build_prompt(last_step)

        # Thought prompting step
        prompt.append({"role": "user", "content": self.thought_prompt})

        thought_prompt_num = 1
        for i in range(self.max_retries):
            llm_response = self.llm_manager.call_llm(prompt, None)

            if llm_response.error: # most likely token limit
                prompt = prompt[0:1] + prompt[(i+1)*3+1:]
                continue
            
            if self._is_valid_thought(llm_response.response.content):
                thought = llm_response.response.content
                print(f"### Thought ### \n{thought}\n")
                self.memory.add_agent_thought(thought)
                prompt = prompt[:-thought_prompt_num] # remove thought prompts
                prompt.append({"role": "assistant", "content": thought})
                break

            prompt.append(llm_response.reponse)    
            prompt.append({"role": "user", "content": self.thought_reprompt}) #add reprompt
            thought_prompt_num += 2


        # Tool calling step
        for _ in range(self.max_retries):
            llm_response = self.llm_manager.call_llm(prompt, self.tools)

            if llm_response.error: # most likely token limit
                print(llm_response.response)
                continue

            tool_calls = llm_response.response.tool_calls
            if tool_calls is not None and len(tool_calls) == 1:
                print(f"### Tool Call ### \n{tool_calls[0].function}\n")
                self.memory.add_agent_tool_call(llm_response.response)
                action = tool_calls[0].function.name
                inputs = json.loads(tool_calls[0].function.arguments)
                return action, inputs 
            else:
                print("No tool call or more than one tool call. Calling LLM again")
                print(llm_response.response)
                prompt.append({"role": "user", "content": "Please select a single tool call"})

        return None, None 

    @abstractmethod
    def _is_valid_thought(self, response):
        pass

    def add_observation(self, observation):
        self.memory.add_env_step(observation)

    def _build_prompt(self, last_step):
        prompt = [{"role": "system", "content": self.system_prompt}]

        prompt += self.memory.retrieve_memory()
                
        if last_step:
            prompt.append({"role": "user", "content": "This is the last step. Please submit the final answer."})

        return prompt


class ReActAgent(Agent):
    def __init__(self, env, llm_manager, memory, X, metadata, max_retries=3) -> None:
        super().__init__(env, llm_manager, memory, X, metadata, max_retries)

        self.thought_prompt = prompts.react_prompt
        self.thought_reprompt = prompts.react_reprompt

    def _is_valid_thought(self, response):
        return True

class PlanningAgent(Agent):
    def __init__(self, env, llm_manager, memory, X, metadata, max_retries=3) -> None:
        super().__init__(env, llm_manager, memory, X, metadata, max_retries)

        self.thought_prompt = prompts.planning_prompt
        self.thought_reprompt = prompts.planning_reprompt

    def _is_valid_thought(self, response):
        return True


class MLAgentBenchAgent(Agent):
    def __init__(self, env, llm_manager, memory, X, metadata, max_retries=3) -> None:
        super().__init__(env, llm_manager, memory, X, metadata, max_retries)

        self.thought_prompt = prompts.MLAgentBench_prompt
        self.thought_reprompt = prompts.MLAgentBench_reprompt

    def _is_valid_thought(self, response):
        # TODO: do parsing
        return True

