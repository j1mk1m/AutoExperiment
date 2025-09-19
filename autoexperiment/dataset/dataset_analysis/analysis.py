import json
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import ast
this_dir = os.path.dirname(__file__)

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