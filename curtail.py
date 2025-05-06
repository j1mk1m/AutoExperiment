import os
import json
import csv
import shutil
import wandb
import argparse
import subprocess
import selectors
import time
import datetime


from litellm import completion, completion_cost
from dataset.dataset import get_datapoint


def get_code_examples(file_path, code_by_id=None):
    if code_by_id is None:
        code_by_id = {}
    print(f"Reading file {file_path}")
    with open(file_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            combined_id = row['combined_id']
            if combined_id not in code_by_id:
                code_by_id[combined_id] = []
            code_by_id[combined_id].append(row['generated_code'])
    return code_by_id


def get_gold_code(file_path, code_by_id=None):
    if code_by_id is None:
        code_by_id = {}
    with open(file_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            combined_id = row['combined_id']
            if combined_id not in code_by_id:
                code_by_id[combined_id] = row["gold_code"]
    return code_by_id


def collect_examples(threshold=10):
    trace_dir = os.path.join("agents", "traces", "full")
    correct_path = os.path.join(trace_dir, "correct", "analysis.csv")
    incorrect_path = os.path.join(trace_dir, "incorrect", "analysis.csv")

    correct_examples = get_code_examples(correct_path)
    incorrect_examples = get_code_examples(incorrect_path)
    gold_code = get_gold_code(correct_path)
    gold_code = get_gold_code(incorrect_path, gold_code)

    trace_dir = os.path.join("agents", "traces", "oracle")
    correct_path = os.path.join(trace_dir, "correct", "analysis.csv")
    incorrect_path = os.path.join(trace_dir, "incorrect", "analysis.csv")

    correct_examples = get_code_examples(correct_path, code_by_id=correct_examples)
    incorrect_examples = get_code_examples(incorrect_path, code_by_id=incorrect_examples)
    gold_code = get_gold_code(correct_path, gold_code)
    gold_code = get_gold_code(incorrect_path, gold_code)

    result = []
    
    for combined_id in gold_code.keys():
        if combined_id not in correct_examples.keys() or combined_id not in incorrect_examples: continue 
        print(f"{combined_id} has {len(correct_examples[combined_id])} correct and {len(incorrect_examples[combined_id])} incorrect examples")
        # if len(correct_examples[combined_id]) >= threshold and len(incorrect_examples[combined_id]) >= threshold:
            # print(combined_id)
        result.append({"combined_id": combined_id, "gold_code": gold_code[combined_id], "correct": correct_examples[combined_id], "incorrect": incorrect_examples[combined_id]})
    
    #Write results to jsonl file
    output_file = os.path.join("agents", "traces", "generated_functions.jsonl")
    with open(output_file, "w") as f:
        for item in result:
            f.write(json.dumps(item) + "\n")
    print(f"Wrote {len(result)} examples to {output_file}")
    return result


def command_line(command, cur_dir):
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True, cwd=cur_dir)

        stdout_lines = []
        stderr_lines = []
        lines = []

        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ)
        selector.register(process.stderr, selectors.EVENT_READ)

        timeout = 60 * 10 # 20 minutes
        start_time = time.time()
        timed_out = False
        while process.poll() is None and selector.get_map():
            elapsed_time = time.time() - start_time
            if elapsed_time >= timeout:
                process.kill()
                timed_out = True
                break

            remaining = max(0.1, min(1.0, timeout - elapsed_time))
            events = selector.select(timeout=remaining)


            for key, _ in events:
                line = key.fileobj.readline()
                if key.fileobj == process.stdout:
                    stdout_lines.append(line)
                    print(line)
                    lines.append(line)
                else:
                    stderr_lines.append(line)
                    print(line)
                    lines.append(line)

        for line in process.stdout:
            stdout_lines.append(line)
            lines.append(line)
        for line in process.stderr:
            stderr_lines.append(line)
            lines.append(line)
        
        selector.close()

        return_code = process.returncode

        if timed_out:
            observation = "".join(stdout_lines) + f"\nProcess timed out after {timeout} seconds"
        elif return_code != 0:
            observation = "".join(stderr_lines)
        else:
            observation = "".join(stdout_lines)

        if observation == "":
            observation = "".join(lines)
        
        return observation
    except Exception as e:
        return f"Something went wrong in executing {command}: {e}."


def run_one(combined_id, code_snippet):
    paper_id, func_id = combined_id.split("_")
    # Get experiments that depend on this function
    exp_file = os.path.join("dataset", "MLRC", "mlrc_long_exps.csv")
    with open(exp_file, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        experiments = [row for row in reader]
    
    experiments = [exp for exp in experiments if exp["paper_id"] == paper_id and func_id in exp["func_dependencies"].split(",")]
    command_sequence = ""
    for i, exp in enumerate(experiments):
        command_sequence += f"echo Experiment {i + 1}\n"+ exp["command_sequence"] + "\n"

    X, y, metadata = get_datapoint(combined_id=combined_id)

    # Read current file
    script_path = os.path.join(X["path"], X['funcs_to_block'][0]['file'])
    with open(script_path, 'r') as f:
        lines = list(f.readlines())
    
    # Get the function lines
    header_line = X['funcs_to_block'][0]["header_line"]
    start = X['funcs_to_block'][0]["line_start"]
    end = X['funcs_to_block'][0]["line_end"]
    
    lines = lines[:header_line-1] + [line + "\n" for line in code_snippet.split("\n")] + lines[end+1:]
    
    # Write back to file
    with open(script_path, 'w') as f:
        f.writelines(lines)
    
    # Replace refsol.sh with command sequence
    refsol_path = os.path.join(X["path"], "refsol.sh")
    with open(refsol_path, 'w') as f:
        f.write(command_sequence)

    # Run refsol.sh
    observation = command_line("bash refsol.sh", X["path"])
    print(observation)

    # Clean up
    shutil.rmtree(X["path"])
    return observation


def main(combined_id):
    output_file = os.path.join("agents", "traces", "generated_functions.jsonl")
    with open(output_file, "r") as f:
        funcs = [json.loads(line) for line in f]

    # Find the function with matching combined_id
    func = None
    for f in funcs:
        if f["combined_id"] == combined_id:
            func = f
            break
            
    if func is None:
        raise ValueError(f"Could not find function with combined_id {combined_id}")

    gold_code = func["gold_code"]
    correct_codes = func["correct"]
    incorrect_codes = func["incorrect"]

    observations = {"correct": [], "incorrect": []}

    # Run oracle
    print("#" * 30)
    print("Oracle")
    print("#" * 30)
    observation = run_one(combined_id, gold_code)
    observations["gold_code"] = observation
    
    # Run correct generations
    for i, code in enumerate(correct_codes):
        print("#" * 30)
        print(f"Correct code {i}")
        print("#" * 30)
        obs = run_one(combined_id, code)
        observations["correct"].append(obs)
    
    # Run incorrect generations
    for i, code in enumerate(incorrect_codes):
        print("#" * 30)
        print(f"Inorrect code {i}")
        print("#" * 30)
        obs = run_one(combined_id, code)
        observations["incorrect"].append(obs)
    
    # Save observations to a file
    output_path = os.path.join("agents", "traces", f"curtail_{combined_id}.json")
    with open(output_path, "w") as f:
        json.dump(observations, f, indent=4)
    print(f"Saved observations to {output_path}")
 
 
if __name__=="__main__":
    # collect_examples()

    parser = argparse.ArgumentParser()
    parser.add_argument("--_tags", type=str, default="curtail,test")
    parser.add_argument("--combined-id", type=str, default="0000.00000_0", help="combined_id = paper_id + func_id")

    args = parser.parse_args()

    tags = ["curtail", args.combined_id]

    wandb.init(
        project="AutoExperiment",
        entity="j1mk1m",
        tags=tags
    )

    main(args.combined_id)


