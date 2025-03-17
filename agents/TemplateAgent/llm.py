from litellm import completion, completion_cost


class LLMResponse:
    def __init__(self, prompt, content, tool_calls, prompt_tokens, completion_tokens, cost) -> None:
        self.prompt = prompt
        self.content = content
        self.tool_calls = tool_calls
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.cost = cost



def call_llm(messages, tools, model):
    try:
        response = completion(model=model, messages=messages, tools=tools)
        cost = completion_cost(completion_response=response, model=model, messages=messages)
        return LLMResponse(prompt=messages,
                           content=response.choices[0].message.content, 
                           tool_calls=response.choices[0].message.tool_calls, 
                           prompt_tokens=response.usage.prompt_tokens, 
                           completion_tokens=response.usage.completion_tokens, 
                           cost=cost)
    except Exception as e:
        message = f"Error calling llm {model}: {e}"
        print(message)
        return LLMResponse(prompt=messages, content=message, tool_calls=None, prompt_tokens=0, completion_tokens=0, cost=0)

