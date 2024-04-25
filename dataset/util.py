import json
import os
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
import ast
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
            lines = file.readlines()
        code = "\n".join(lines[func["line_start"]-1:func["line_end"]])
        message = SCRIPT.format(code=code)
        response = call_openai([{"role": "user", "content": message}], None, "gpt-4-1106-preview")
        print(response.content)
        func["description"] = response.content.split('"""')[1].strip()
    with open(os.path.join(this_dir, paper_id, "functions.json"), 'w') as file:
        json.dump(contents, file, indent=4)

def calculate_stats():
    exp_counts = {}
    with open(os.path.join(this_dir, "experiment_csvs", "experiments-full.csv"), 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            paper_id = row["paper_id"]
            if paper_id not in exp_counts:
                exp_counts[paper_id] = 0
            exp_counts[paper_id] += 1                
    total_lines = 0
    paper_count = 0
    func_count = 0
    lines = []
    class_func = 0
    func_calls = []
    datapoint_count = 0
    for dir in os.listdir(this_dir):
        if dir[0].isnumeric():
            json_path = os.path.join(this_dir, dir, "functions.json")
            if not os.path.exists(json_path): continue
            print("Parsing", dir)

            paper_count += 1
            with open(json_path, 'r') as file:
                content = json.loads(file.read())
            for func in content["functions"]:
                num_lines = func["line_end"] - func["line_start"]
                total_lines += num_lines
                lines.append(num_lines)
                func_count += 1
                if func["class"] == "True":
                    class_func += 1
                with open(os.path.join(this_dir, dir, "code", func["script"]), 'r') as file:
                    contents = file.readlines()
                contents = contents[func["line_start"]-1:func["line_end"]]
                first = contents[0].replace("\t", "    ")
                c = 0
                while first[c].isspace():
                    c += 1
                contents = "".join([line.replace("\t", "    ")[c:] for line in contents])
                ast_string = ast.dump(ast.parse(contents), indent=4)
                func_calls.append(ast_string.count("Call"))
            datapoint_count += len(content["functions"]) * exp_counts[dir]
    print("Average num lines", total_lines / func_count)
    print("Average func per paper", func_count / paper_count)
    print("Total functions removed", func_count)
    print("Class functions", class_func, "Standalone", func_count - class_func)
    print("Average Function call", sum(func_calls) / func_count)
    print("Total number of PC datapoints", datapoint_count)

    # Distribution of func calls
    # print(func_calls)
    # n, bins, patches = plt.hist(func_calls)
    # plt.figure(figsize=(10, 6))
    # plt.hist(func_calls, bins=20, color='skyblue', edgecolor='black')
    # plt.xlabel('Number of Function Calls', fontsize=18)
    # plt.ylabel('Frequency', fontsize=18)
    # plt.grid(axis='y', alpha=0.75)
    # plt.savefig("num_func_calls.png")

    # Class vs Standalone
    # plt.figure(figsize=(10, 6))
    # labels = 'Class', 'Standalone'
    # sizes = [class_func, func_count - class_func]
    # fig, ax = plt.subplots()
    # ax.pie(sizes, labels=labels, autopct='%1.1f%%', textprops={'fontsize': 18})
    # plt.savefig("class_standalone.png")

    # Distribution of Num Lines
    # print(lines)
    # n, bins, patches = plt.hist(lines)
    # plt.figure(figsize=(10, 6))
    # plt.hist(lines, bins=20, color='skyblue', edgecolor='black')
    # plt.xlabel('Number of Lines', fontsize=18)
    # plt.ylabel('Frequency', fontsize=18)
    # plt.grid(axis='y', alpha=0.75)
    # plt.savefig("num_lines_hist.png")

calculate_stats()
"""
papers = ["2307.03135"]
for paper in papers:
    generate_code_comments(paper)
"""