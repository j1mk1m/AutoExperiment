from litellm import completion, completion_cost


class LLMResponse:
    def __init__(self, prompt, response, prompt_tokens, completion_tokens, cost, error) -> None:
        self.prompt = prompt
        self.response = response
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.cost = cost
        self.error = error


class LLM:
    def __init__(self, model_backbone) -> None:
        self.cost = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.model_backbone = model_backbone

    def call_llm(self, messages, tools, model=None):
        if model is None:
            model = self.model_backbone

        response = call_llm(messages, tools, model)

        self.cost += response.cost
        self.prompt_tokens += response.prompt_tokens
        self.completion_tokens += response.completion_tokens
        return response


def call_llm(messages, tools, model):
    try:
        response = completion(model=model, messages=messages, tools=tools)
        cost = completion_cost(completion_response=response, model=model, messages=messages)
        return LLMResponse(prompt=messages,
                        response=response.choices[0].message, 
                        prompt_tokens=response.usage.prompt_tokens, 
                        completion_tokens=response.usage.completion_tokens, 
                        cost=cost,
                        error=False)
    except Exception as e:
        message = f"Error calling llm {model}: {e}"
        print(message)
        return LLMResponse(prompt=messages, response=message, prompt_tokens=0, completion_tokens=0, cost=0, error=True)

