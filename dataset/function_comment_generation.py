import json
import os

this_dir = os.path.dirname(__file__)

SCRIPT = """
Given this Python function, generate a Python docstring that describes the following:
- simple one-line description of the function
- arguments that the function takes in and their types
- return value(s) and their types

Here is an example
Python function: 
def get_scaled_mask(self, mask, size, epsilon):
    if isinstance(size, int):
        size = (size, size)
    s = transforms.ToTensor()(mask).to(self.device)
    s = transforms.Resize(size=size)(s)
    s = (s > 0).to(torch.float32)
    s = torch.where(s == 0, epsilon, s)
    return s

Docstring:
\"\"\"
Returns a scaled mask of the same size as the input mask 
:param size: An integer or a tuple 
:param mask: A tensor 
:return: A tensor of type float32
\"\"\"

Now let's try
Python function:
{code}

Docstring:
"""

OLD_SCRIPT = """
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


def generate_code_comments(paper_id, split="MLRC"):
    print(f"Paper ID: {paper_id}")
    with open(os.path.join(this_dir, split, paper_id, "all_functions.jsonl"), 'r') as file:
        functions = [json.loads(line) for line in file]

    for i, func in enumerate(functions):
        if "description" in func:
            continue
        print(f"Generating comments for {func['name']}")
        with open(os.path.join(this_dir, "MLRC", paper_id, "code", func["file"]), 'r') as file:
            lines = file.readlines()
        code = "\n".join(lines[int(func["line_start"])-1:int(func["line_end"])])

        message = SCRIPT.format(code=code)
        response = call_openai([{"role": "user", "content": message}], None, "gpt-4o-mini")
        print(response.content)

        func["description"] = response.content.split('"""')[1].strip()

        if (i+1) % 100 == 0:
            print(f"Saving {i+1} functions")
            with open(os.path.join(this_dir, split, paper_id, "all_functions.jsonl"), 'w') as file:
                for func in functions:
                    json.dump(func, file)
                    file.write('\n')

    with open(os.path.join(this_dir, split, paper_id, "all_functions.jsonl"), 'w') as file:
        for func in functions:
            json.dump(func, file)
            file.write('\n')

papers = ["2309.05569"]
print("Generating Comments for ", papers)
for paper in papers:
    generate_code_comments(paper)