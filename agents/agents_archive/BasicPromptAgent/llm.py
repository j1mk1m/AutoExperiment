import os
import tiktoken
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def call_openai(messages, tools, model):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools
        )
        return response.choices[0].message
    except Exception as e:
        print(e)

def call_llm(messages, tools=None, model="gpt-4o"):
    if 'gpt' in model:
        return call_openai(messages, tools, model)
    else:
        raise NotImplemented

