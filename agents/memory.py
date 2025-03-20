from abc import ABC, abstractmethod 

def add_memory_args(parser):
    parser.add_argument("--memory", type=str, choices=["Full", "SlidingWindow", "RAG", "HippoRAG"], required=True,
                        help="Type of memory to use")
    parser.add_argument("--lookback", type=int, default=5,
                        help="Number of previous steps to include in sliding window memory")

class Memory(ABC):
    def __init__(self, **kwargs) -> None:
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
    def __init__(self, lookback, **kwargs) -> None:
        super().__init__(**kwargs)
        self.lookback = lookback
    
    def retrieve_memory(self):
        prompt = []
        
        len_obs = len(self.observations)
        for i in range(max(0, len_obs - self.lookback), len_obs):
            prompt.append({"role": "assistant", "content": self.thoughts[i]})
            prompt.append(self.tool_calls[i])
            prompt.append(self.observations[i])
        
        return prompt


