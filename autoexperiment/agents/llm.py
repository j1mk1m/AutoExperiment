from litellm import completion, completion_cost
import openai
import asyncio
import os
from sshtunnel import SSHTunnelForwarder
import asyncssh
from tqdm import tqdm


INFERENCE_PORT = 8084
VLLM_SERVER_NAME = "shire-1-10"

client = openai.AsyncOpenAI(api_key='EMPTY', base_url=f'http://localhost:{INFERENCE_PORT}/v1')
SSH_KEY = os.path.expanduser('~/.ssh/id_rsa')

async def fetch_completion(client, msgs, tools, sampling_params):
    return await client.chat.completions.create(
        messages=msgs,
        model=sampling_params["model"],
        n=sampling_params["k"],
        temperature=sampling_params["temperature"],
        max_tokens=sampling_params["max_tokens"],
        tools=tools,
    )

async def inference_async(all_msgs, tools, sampling_params, vllm_server_name):
    async with asyncssh.connect(
        "babel.lti.cs.cmu.edu",
        username='gyeongwk',
        port=22,
        client_keys=[SSH_KEY],
        known_hosts=None,
        # keepalive_interval=60, 
        # keepalive_count_max=3,
    ) as conn:
        listener = await conn.forward_local_port('127.0.0.1', INFERENCE_PORT, vllm_server_name, INFERENCE_PORT)
        local_port = listener.get_port()
        # print(f"Tunnel up: 127.0.0.1:{local_port}")

        # Create one task per msgs
        tasks = [fetch_completion(client, msgs, tools, sampling_params) for msgs in all_msgs]

        # Gather results concurrently
        responses = await asyncio.gather(*tasks)
        return responses

def inference(all_msgs, tools, sampling_params, batch_size):
    """
    Run inference over `all_msgs` in sequential batches of size `batch_size`,
    displaying a progress bar. Results are returned in the original order.
    """
    results = []
    total = len(all_msgs)
    # for i in tqdm(range(0, total, batch_size), desc=f"Inference across batches of size {batch_size}", unit="batch"):
    for i in range(0, total, batch_size):
        batch = all_msgs[i:i + batch_size]
        batch_res = asyncio.run( inference_async(batch, tools, sampling_params, VLLM_SERVER_NAME) )
        results.extend(batch_res)
    return results


class LLMResponse:
    def __init__(self, prompt, response, prompt_tokens, completion_tokens, cost, error) -> None:
        self.prompt = prompt
        self.response = response
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.cost = cost
        self.error = error


class LLM:
    def __init__(self, model_backbone, compute_budget=None) -> None:
        self.cost = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

        self.compute_budget = compute_budget
        self.model_backbone = model_backbone

    def is_over_compute_budget(self):
        return self.compute_budget is not None and self.cost > self.compute_budget

    def call_llm(self, messages, tools, model=None, **kwargs):
        if model is None:
            model = self.model_backbone

        if self.compute_budget is not None and self.cost > self.compute_budget:
            return LLMResponse(messages, response=f"Exceeded compute budget of {self.compute_budget}", prompt_tokens=0, completion_tokens=0, cost=0, error=True)

        response = call_llm(messages, tools, model, **kwargs)

        self.cost += response.cost
        self.prompt_tokens += response.prompt_tokens
        self.completion_tokens += response.completion_tokens
        return response


def call_llm(messages, tools, model, **kwargs):
    if "vllm" in model:
        model_name = model.split(":")[1]
        sampling_params = {
            "model": model_name,
            "k": kwargs.get("k", 1),
            "temperature": kwargs.get("temperature", 0.0),
            "max_tokens": kwargs.get("max_tokens", 1000),
        }
        responses = inference([messages], tools, sampling_params, batch_size=1)
        return LLMResponse(prompt=messages,
                    response=responses[0].choices[0].message, 
                    prompt_tokens=responses[0].usage.prompt_tokens, 
                    completion_tokens=responses[0].usage.completion_tokens, 
                    cost=0,
                    error=False)

    try:
        
        if tools is None:
            response = completion(model=model, messages=messages, **kwargs)
        else:
            response = completion(model=model, messages=messages, tools=tools, **kwargs)
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

