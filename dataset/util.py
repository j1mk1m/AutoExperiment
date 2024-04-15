import json
import os
import sys
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

def generate_code_comments(paper_id):
    with open(os.path.join(this_dir, paper_id, "functions.json"), 'r') as file:
        contents = json.loads(file.read())
    for func in contents["functions"]:
        print(f"Generating comments for {func['name']}")
        with open(os.path.join(this_dir, paper_id, "code", func["script"]), 'r') as file:
            code = file.read()
        code = "def " + func["name"] + code.split("def " + func["name"])[1].split("def ")[0]
        message = SCRIPT.format(code=code)
        response = call_openai([{"role": "user", "content": message}], None, "gpt-4-1106-preview")
        print(response.content)
        func["description"] = response.content.split('"""')[1].strip()
    with open(os.path.join(this_dir, paper_id, "functions.json"), 'w') as file:
        json.dump(contents, file, indent=4)

generate_code_comments("0000.00000")
