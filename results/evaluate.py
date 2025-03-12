from losses import *
import os
import json
import csv
this_dir = os.path.dirname(__file__)

file = os.path.join(this_dir, "autoagent_data.csv")

with open(file, 'r') as f:
    reader = csv.DictReader(f)
    rows = list(reader)

for row in rows:
    agent_output = row["agent_output"].replace("'", "\"")
    gold_output = row["gold_output"].replace("'", "\"")
    try:
        agent_output = json.loads(agent_output) if agent_output != "" else {}
    except Exception as e:
        print(e)
        print(agent_output)
        agent_output = {}
    try:
        gold_output = json.loads(gold_output)
    except Exception as e:
        print(e)
        print(gold_output)
        raise Exception

    loss_per_exp, correct_per_exp, correct_count, all_correct = calculate_loss(gold_output, agent_output)
    row["loss_per_exp"] = loss_per_exp
    row["correct_per_exp"] = correct_per_exp
    row["correct_count"] = correct_count
    row["all_correct"] = all_correct

# Write updated rows back to CSV
output_file = os.path.join(this_dir, "autoagent_evaluated_data.csv")
with open(output_file, 'w', newline='') as f:
    # Get all fieldnames from first row
    fieldnames = list(rows[0].keys())
    
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)


