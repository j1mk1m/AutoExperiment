import json
import csv 
import pandas as pd
from scipy.stats import chi2

from paired_bootstrap import eval_with_paired_bootstrap

# McNemar's test

def mcnemar_test(sys1, sys2):
    a, b, c, d = 0, 0, 0, 0
    for i in range(len(sys1)):
        if sys1[i] == 1 and sys2[i] == 1:
            a += 1
        elif sys1[i] == 1 and sys2[i] == 0:
            b += 1
        elif sys1[i] == 0 and sys2[i] == 1:
            c += 1
        elif sys1[i] == 0 and sys2[i] == 0:
            d += 1

    chi_squared = ((abs(a - d) - 1) ** 2) / (a + d)
    p_value = 1 - chi2.cdf(chi_squared, 1)

    print(f"p-value: {p_value:.4f}")
    if p_value < 0.05:
        print("The difference is statistically significant (p < 0.05)")
    else:
        print("The difference is not statistically significant (p >= 0.05)")
    return p_value

def load_data(file_path):
    with open(file_path, 'r') as file:
        reader = csv.DictReader(file)
        data = list(reader)
    results = {}
    for row in data:
        results[row["combined-id"]] = row["all_correct"]
    return results

def load_csv(file_path):
    return pd.read_csv(file_path)

def convert_to_binary(data):
    return [1 if result == "TRUE" or result == "true" else 0 for result in data]

def convert_dict_to_binary(data1, data2):
    sys1 = []
    sys2 = []
    for combined_id, result1 in data1.items():
        if combined_id in data2:
            result2 = data2[combined_id]
            sys1.append(1 if result1 == "TRUE" or result1 == "true" else 0)
            sys2.append(1 if result2 == "TRUE" or result2 == "true" else 0)
    return sys1, sys2
    

def main_results():
    files = {
        # "GPT-4o": "dynamic_gpt_4o.csv",
        # "GPT-4o-mini": "dynamic_gpt_4o_mini.csv",
        # "Claude-3.5-sonnet": "dynamic_claude_3_5.csv",
        # "Claude-3.7-sonnet": "dynamic_claude_3_7.csv",
        # "GPT-4o": "gpt_4o_n_2.csv",
        # "GPT-4o-mini": "gpt_4o_mini_n_2.csv",
        # "Claude-3.5-sonnet": "claude_3_5_n_2.csv",
        # "Claude-3.7-sonnet": "claude_3_7_n_2.csv",
        "GPT-4o": "n_4_gpt_4o.csv",
        "GPT-4o-mini": "n_4_gpt_4o_mini.csv",
        "Claude-3.5-sonnet": "n_4_claude_3_5.csv",
        "Claude-3.7-sonnet": "n_4_claude_3_7.csv",
    }

    data = {}
    for model, file in files.items():
        data[model] = load_data("data/" + file)


    # Perform McNemar's test for each pair of models
    models = list(data.keys())
    for i in range(len(models)):
        for j in range(i+1, len(models)):
            print("########################")
            model1 = models[i]
            model2 = models[j]

            # sys1, sys2 = convert_dict_to_binary(data[model1], data[model2])
            sys1 = [0 for _ in range(37)]
            sys2 = [0 for _ in range(37)]
            gold = [1 for _ in range(len(sys1))]
            
            # print(f"\nMcNemar's test results for {model1} vs {model2}:")
            # p_value = mcnemar_test(sys1, sys2)  
            
            print(f"\nPaired bootstrap results for {model1} vs {model2}:")
            eval_with_paired_bootstrap(gold, sys1, sys2)
        

def verifier_results(data):
    pass_1 = convert_to_binary(data["Run 1"])
    pass_5 = convert_to_binary(data["Pass@5"])
    verifier = convert_to_binary(data["Verifier Correct"])

    gold = [1 for _ in range(len(pass_1))]

    print(f"\nPaired bootstrap results for pass@1 and pass@5:")
    eval_with_paired_bootstrap(gold, pass_1, pass_5)

    print(f"\nPaired bootstrap results for pass@1 and verifier:")
    eval_with_paired_bootstrap(gold, pass_1, verifier)

    print(f"\nPaired bootstrap results for pass@5 and verifier:")
    eval_with_paired_bootstrap(gold, pass_5, verifier)


def run_verifier_results():
    gpt_data = load_csv("data/verifier_gpt4o.csv")
    claude_data = load_csv("data/verifier_claude_3_5.csv")
    print("GPT-4o")
    verifier_results(gpt_data)
    print("Claude 3.5")
    verifier_results(claude_data)


def agent_architecture_results():
    react_full = load_data("data/react_full.csv")
    react_sliding = load_data("data/react_sliding.csv")
    mla_sliding1 = load_data("data/mla_sliding1.csv")
    mla_full_mini = load_data("data/mla_full_mini.csv")

    sys1, sys2 = convert_dict_to_binary(react_full, react_sliding)
    gold = [1 for _ in range(len(sys1))]
    print(f"\nPaired bootstrap results for react_full and react_sliding:")
    eval_with_paired_bootstrap(gold, sys1, sys2)

    sys1, sys2 = convert_dict_to_binary(react_full, mla_sliding1)
    gold = [1 for _ in range(len(sys1))]
    print(f"\nPaired bootstrap results for react_full and mla_sliding1:")
    eval_with_paired_bootstrap(gold, sys1, sys2)

    sys1, sys2 = convert_dict_to_binary(react_sliding, mla_sliding1)
    gold = [1 for _ in range(len(sys1))]
    print(f"\nPaired bootstrap results for react_sliding and mla_sliding1:")
    eval_with_paired_bootstrap(gold, sys1, sys2)

    sys1, sys2 = convert_dict_to_binary(react_full, mla_full_mini)
    gold = [1 for _ in range(len(sys1))]
    print(f"\nPaired bootstrap results for react_full and mla_full_mini:")
    eval_with_paired_bootstrap(gold, sys1, sys2)

def run_nl_results():
    data = load_csv("n_2.csv")

    sys1 = list(data["No 1"]) + list(data["No 2"]) + list(data["No 3"]) #+ list(data["no 4"]) + list(data["no 5"])
    # sys1 = list(data["no 1"]) + list(data["no 2"]) + list(data["no 3"]) + list(data["no 4"]) + list(data["no 5"])
    sys2 = list(data["Full 1"]) + list(data["Full 2"]) + list(data["Full 3"]) #+ list(data["full 4"]) + list(data["full 5"])
    # sys2 = list(data["full 1"]) + list(data["full 2"]) + list(data["full 3"]) + list(data["full 4"]) + list(data["full 5"])
    # sys1 = [1 if result else 0 for result in sys1]
    sys1 = [1 if result=="x" else 0 for result in sys1]
    # sys2 = [1 if result else 0 for result in sys2]
    sys2 = [1 if result=="x" else 0 for result in sys2]
    gold = [1 for _ in range(len(sys1))]
    print(sys1)


    print(f"\nPaired bootstrap results for no and full:")
    eval_with_paired_bootstrap(gold, sys1, sys2)



# run_verifier_results()
# agent_architecture_results()
# main_results()
run_nl_results()



                