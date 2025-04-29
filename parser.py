import os
import json
import shutil
from litellm import completion, completion_cost
from dataset.dataset import get_datapoint

class LLMResponse:
    def __init__(self, prompt, response, prompt_tokens, completion_tokens, cost, error) -> None:
        self.prompt = prompt
        self.response = response
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.cost = cost
        self.error = error

this_dir = os.path.dirname(__file__)

def percent_loss(gold, pred):
    return abs(gold - pred) / gold

def collect_loss(gold, pred, metric_fn=percent_loss):
    loss = []
    if isinstance(pred, str):
        try:
            pred = json.loads(pred)
        except Exception as e:
            try:
                pred = int(pred)
            except Exception as e:
                # print(f"Error in converting prediction: {pred}")
                return [1]

    if isinstance(pred, dict):
        for key in gold:
            if key not in pred:
                pred[key] = 0.0
            loss += collect_loss(gold[key], pred[key], metric_fn)
    elif isinstance(pred, list):
        for i in range(len(pred)):
            loss += collect_loss(gold[i], pred[i], metric_fn)
    else:
        try:
            loss += [metric_fn(gold, pred)]
        except Exception as e:
            # print(f"Got error during loss calculation: {e}")
            return [1]
    return loss


def calculate_loss(gold, pred, metric_fn=percent_loss):
    try:
        if isinstance(pred, str):
            pred = json.loads(pred)

        loss_per_exp = {}
        correct_per_exp = {}
        correct_count = 0
        all_correct = True
        
        for key in gold:
            if key in pred:
                losses = collect_loss(gold[key], pred[key], metric_fn)
                loss = sum(losses) / len(losses)
            else:
                loss = 1
            loss_per_exp[key] = loss
            correct = loss <= 0.05
            if correct:
                correct_count += 1 
            correct_per_exp[key] = correct
            all_correct = all_correct and correct
        return loss_per_exp, correct_per_exp, correct_count, all_correct
    except Exception as e:
        # print(e)
        return None, None, 0, False
 

def get_gold_output_and_code(combined_id):
    X, y, metadata = get_datapoint(combined_id=combined_id)
    gold_code = X["funcs_to_block"][0]["gold_standard_code"]
    shutil.rmtree(X["path"])
    return gold_code, y
    
def call_llm(messages, model, **kwargs):
    try:
        response = completion(model=model, messages=messages)
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


def run_analysis(contents, gold_code):
    # extract all the diffs
    diffs = contents.split("+++")[1:]
    diffs = [diff.split("######################################")[0] for diff in diffs]
    diffs = '\n'.join(diffs)
    # compare 
    prompt = f"""I have a sequence of diffs from a coding agent and the oracle code that it is trying to generate. The agent-generated code runs but produces an incorrect result. Explain why the agent code does not work.

Diff:
{diffs}

Oracle Code:
{gold_code}
"""
    
    # print("#" * 10 + "PROMPT" + "#"*10)
    # print(prompt)

    response = call_llm([{"role": "user", "content": prompt}], "gpt-4o-mini")
    print("#" * 10 + " RESPONSE " + "#"*10)
    print(response.response.content)
    print("#" * 30)
    return response.response.content


def main():
    # Get all files in agent_architecture directory
    arch_dir = os.path.join("agents", "logs", "agent_architecture", "gpt-4o")
    results = []
    
    # Walk through all files in directory
    for root, dirs, files in os.walk(arch_dir):
        for filename in files:
            filepath = os.path.join(root, filename)
            
            # Read contents of each file
            try:
                with open(filepath, 'r') as f:
                    contents = f.read()
                results.append({
                    'filename': filename,
                    'contents': contents
                })

                combined_id = "_".join(filename.split("agent_architecture_")[1].split("_")[0:2])

                if "2110." not in combined_id: continue

                final_answer = contents.split("Observation:\n")[-1].split("\n")[0]
                final_answer = json.loads(final_answer)

                gold_code, gold_output = get_gold_output_and_code(combined_id)

                loss_per_exp, correct_per_exp, correct_count, all_correct = calculate_loss(gold_output, final_answer)
                if any([loss != 0.0 and loss != 1.0 for loss in loss_per_exp.values()]) and not all_correct:
                    print(f"Found example at {filename}")
                    run_analysis(contents, gold_code)

            except Exception as e:
                pass
                

if __name__=="__main__":
    main()