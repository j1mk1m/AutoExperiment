import json
import os
# import sys
# import csv
# import numpy as np
# import matplotlib.pyplot as plt
# import ast
this_dir = os.path.dirname(__file__)

SCRIPT = """
Given this Python function, generate a Python docstring that contain all information necessary to rewrite the code. 
You should include the following:
- arguments that the function takes in
- what the function modifies (like globals and class variables)
- effects (like print statements)
- return value(s)

Here is an example
Python function: 
def face_locations(img, number_of_times_to_upsample=1, model="hog"):
    print("Finding face locations")
    self.model = model
    if model == "cnn": 
        return [_trim_css_to_bounds(_rect_to_css(face.rect), img.shape) for face in _raw_face_locations(img, number_of_times_to_upsample, "cnn")] 
    else: 
        return [_trim_css_to_bounds(_rect_to_css(face), img.shape) for face in _raw_face_locations(img, number_of_times_to_upsample, model)]

Docstring:
\"\"\"
Returns an array of bounding boxes of human faces in a image 
:param img: An image (as a numpy array) 
:param number_of_times_to_upsample: How many times to upsample the image looking for faces. Higher numbers find smaller faces. 
:param model: Which face detection model to use. "hog" is less accurate but faster on CPUs. "cnn" is a more accurate deep-learning model which is GPU/CUDA accelerated (if available). The default is "hog". 
:modifies self.model: sets self.model to parameter model
:effects: prints string literal "Finding face locations"
:return: A list of tuples of found face locations in css (top, right, bottom, left) order 
\"\"\"

Now let's try
Python function:
{code}

Docstring:
"""

REVERSE_PROMPT = """
Given a Python function and relevant textual information about how the function is implemented, generate a docstring for the function. 
The docstring should contain enough information such that the Python function is implementable from the docstring.
However, the docstring should only provide information that you cannot obtain from the text. 

Here is an example
Python function:
def fibonacci(n): 
    if n <= 0:
        return [0]
    elif n == 1:
        return [0, 1]
    
    fibonacci = [0, 1]
    for _ in range(2, n + 1):
        fibonacci.append(fibonacci[-2] + fibonacci[-1])
    return fibonacci

Text: We implement a function that produces up to the nth fibonacci number iteratively.

Docstring:
\"\"\"
:param n: The position in the Fibonacci sequence to compute. Must be a non-negative integer.
:return: a Python list containing Fibonacci numbers up to n
\"\"\"

Now let's try
Python function:
{code}

Text:
{text}

Docstring:
"""

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


import json

# Load the super_pc_llama.jsonl file
# super_pc_llama_data = []
# with open('super_pc_llama.jsonl', 'r') as file:
#     super_pc_llama_data = [json.loads(line) for line in file]

def generate_code_comments(paper_id, split="MLRC"):
    print(f"Paper ID: {paper_id}")
    functions = []
    with open(os.path.join(this_dir, split, paper_id, "functions.json"), 'r') as file:
        functions = json.load(file)["functions"]
    for func in functions:
        print(f"Generating comments for {func['name']}")
        with open(os.path.join(this_dir, "MLRC", paper_id, "code", func["file"]), 'r') as file:
            lines = file.readlines()
        code = "\n".join(lines[int(func["line_start"])-1:int(func["line_end"])])
        # matches = [data["paper_content"] for data in super_pc_llama_data if data["script"] == func["script"] and data["function_name"] == func["name"]]
        # if len(matches) == 0:
        #     continue
        # text = matches[0]
        message = SCRIPT.format(code=code)
        # message = REVERSE_PROMPT.format(code=code, text=text)
        response = call_openai([{"role": "user", "content": message}], None, "gpt-4o")
        print(response.content)
        func["description"] = response.content.split('"""')[1].strip()
    with open(os.path.join(this_dir, split, paper_id, "functions_new.jsonl"), 'w') as file:
        for func in functions:
            json.dump(func, file)
            file.write('\n')

# papers = ["2105.14761", "2203.07836", "2309.07045", "2305.15933", "2403.07088", "2305.17333", "2210.14102", "2304.13148", "2401.15535"]
papers = ["2110.03485", "2205.00048", "2303.11932", "2309.05569"]
# papers = ["2303.11932", "2309.05569"]
print("Generating Comments for ", papers)
for paper in papers:
    generate_code_comments(paper)