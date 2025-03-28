from abc import ABC, abstractmethod 

def add_memory_args(parser):
    parser.add_argument("--memory", type=str, choices=["Full", "SlidingWindow", "Summary"], required=True,
                        help="Type of memory to use")
    parser.add_argument("--lookback", type=int, default=5,
                        help="Number of previous steps to include in sliding window memory")

class Memory(ABC):
    def __init__(self, llm_manager, **kwargs) -> None:
        self.llm_manager = llm_manager

        self.thoughts = []
        self.tool_calls = []
        self.observations = []

    def add_agent_thought(self, thought):
        self.thoughts.append(thought)

    def add_agent_tool_call(self, tool_call):
        self.tool_calls.append(tool_call)

    def add_env_step(self, observation):
        tool_call = self.tool_calls[-1].tool_calls[0]
        self.observations.append({"role": "tool", "tool_call_id": tool_call.id, "content": observation})

    @abstractmethod
    def retrieve_memory(self):
        pass

class FullMemory(Memory):
    def retrieve_memory(self):
        prompt = []
        for i in range(len(self.observations)):
            prompt.append({"role": "assistant", "content": self.thoughts[i]})
            prompt.append(self.tool_calls[i])
            prompt.append(self.observations[i])
        
        return prompt

class SlidingWindowMemory(Memory):
    def __init__(self, llm_manager, lookback, **kwargs) -> None:
        super().__init__(llm_manager, **kwargs)
        self.lookback = lookback
    
    def retrieve_memory(self):
        prompt = []
        
        len_obs = len(self.observations)
        for i in range(max(0, len_obs - self.lookback), len_obs):
            prompt.append({"role": "assistant", "content": self.thoughts[i]})
            prompt.append(self.tool_calls[i])
            prompt.append(self.observations[i])
        
        return prompt


class Summary(Memory):
    def __init__(self, llm_manager, lookback, **kwargs) -> None:
        super().__init__(llm_manager, **kwargs)
        self.lookback = lookback
        self.summary = ""

    def add_env_step(self, observation):
        super().add_env_step(observation)

        if len(self.observations) % self.lookback == 0:
            prompt = [{"role": "user", "content": "Given the following agent steps, summarize the important information, making sure to keep precise information."}]

            for i in range(len(self.observations)):
                prompt.append({"role": "assistant", "content": self.thoughts[i]})
                prompt.append(self.tool_calls[i])
                prompt.append(self.observations[i]) 

            self.summary = self.llm_manager.call_llm(prompt, None)


    def retrieve_memory(self):
        prompt = []

        prompt.append({"role": "system", "content": f"Summary of previous steps: {self.summary}"})

        len_obs = len(self.observations)
        for i in range(max(0, len_obs - self.lookback), len_obs):
            prompt.append({"role": "assistant", "content": self.thoughts[i]})
            prompt.append(self.tool_calls[i])
            prompt.append(self.observations[i])
        
        return prompt
