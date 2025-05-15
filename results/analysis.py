import pandas as pd
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# from statsmodels.formula.api import logit


def create_heatmap(df):
    plt.figure(figsize=(10, 6))
    for _, row in df.iterrows():
        line_count = row['line_count']
        func_count = row['func_call_count']
        all_correct = row['all_correct'] 
    
        # Plot checkmark (✓) for correct solutions, x mark (x) for incorrect
        marker = 'o' if all_correct else 'x'  # Using plain 'x' instead of '×'
        color = 'green' if all_correct else 'red'
        plt.scatter(func_count, line_count, marker=marker, c=color, s=100)

    plt.xlabel('Number of Function Calls')
    plt.ylabel('Number of Lines')
    plt.title('Solution Complexity vs Correctness')

    # Add legend
    plt.scatter([], [], marker='o', c='green', s=100, label='Correct')
    plt.scatter([], [], marker='x', c='red', s=100, label='Incorrect')
    plt.legend()
    plt.savefig('heatmap.png')
    plt.close()

def get_step_correct(df):
    # Calculate success rate by step
    step_correct = {}
    for k in range(0, 18, 2):
        if k <= 16:  # Only include steps up to 20
            total_runs = len(df)
            correct_runs = len(df[(df['step'] <= k) & (df['all_correct'])])
            step_correct[k] = correct_runs / total_runs
    print(step_correct)
    return step_correct

def plot_step_correct():
    # Read the CSV file
    df = pd.read_csv("dynamic_o3_mini.csv")
    step_correct_o3_mini = get_step_correct(df)
    df = pd.read_csv("dynamic_gpt_4o.csv")
    step_correct_gpt_4o = get_step_correct(df)
    df = pd.read_csv("dynamic_gpt_4o_mini.csv")
    step_correct_gpt_4o_mini = get_step_correct(df)
    df = pd.read_csv("dynamic_claude_3_5.csv")
    step_correct_claude_3_5 = get_step_correct(df)

    # Plot success rate as a line plot with larger labels and title
    plt.rcParams.update({'font.size': 14})  # Increase base font size
    plt.rcParams['axes.titlesize'] = 16     # Increase title font size
    plt.rcParams['axes.labelsize'] = 14     # Increase axis label font size
    plt.figure(figsize=(10, 6))
    plt.plot(list(step_correct_o3_mini.keys()), list(step_correct_o3_mini.values()), marker='o', label='o3-mini')
    plt.plot(list(step_correct_gpt_4o.keys()), list(step_correct_gpt_4o.values()), marker='o', label='gpt-4o')
    plt.plot(list(step_correct_gpt_4o_mini.keys()), list(step_correct_gpt_4o_mini.values()), marker='o', label='gpt-4o-mini')
    plt.plot(list(step_correct_claude_3_5.keys()), list(step_correct_claude_3_5.values()), marker='o', label='claude-3.5-sonnet')
    plt.xlabel('Max Interactions')
    plt.ylabel('Success Rate')
    plt.title('Performance vs Max Interactions')
    plt.xticks(range(0, 18, 2))  # Set x-axis ticks from 0 to 20 in increments of 2
    plt.ylim(0, 0.5)  # Set y-axis limits from 0 to 1.0
    plt.legend()
    plt.savefig('performance_vs_max_interactions.png')
    plt.close()


def plot_success_across_max_token():
    df = pd.read_csv("fixed_across_max_tokens.csv")
    max_tokens = [tokens for tokens in df.keys() if tokens.isnumeric()]
    success_rates = {"o1": [], "o3-mini": []}
    for token in max_tokens:
        # print(df[token])
        success_rates["o1"].append(df[token][1])
        success_rates["o3-mini"].append(df[token][0])

    plt.figure(figsize=(10, 6))
    plt.plot(max_tokens, success_rates["o1"], marker='o', label='o1')
    plt.plot(max_tokens, success_rates["o3-mini"], marker='o', label='o3-mini')
    plt.xlabel('Max Tokens')
    plt.ylabel('Success Rate')
    plt.title('Performance vs Max Tokens')
    plt.ylim(0, 0.5)  # Set y-axis limits from 0 to 1.0
    plt.legend()
    plt.savefig('performance_vs_max_tokens.png')
    plt.close() 

def plot_main_results():
    df = pd.read_csv("main_results.csv")
    num_removed = df["N"]
    plt.figure(figsize=(10, 6))
    plt.plot(num_removed, df["gpt-4o"], marker='o', label='gpt-4o')
    plt.plot(num_removed, df["gpt-4o-mini"], marker='o', label='gpt-4o-mini')
    plt.plot(num_removed, df["claude-3.5-sonnet"], marker='o', label='claude-3.5-sonnet')
    plt.plot(num_removed, df["claude-3.7-sonnet"], marker='o', label='claude-3.7-sonnet')
    plt.xlabel('n')
    plt.ylabel('Pass Rate (%)')
    plt.legend()
    plt.savefig('main_results.png')
    plt.close()


# plot_step_correct()
# plot_success_across_max_token()
plot_main_results()


### DEPRECATED ###
# Create lists to store successful and failed runs for each metric
# metrics = {
#     'Number of lines': ([], []),
#     'Number test cases': ([], []),
#     'Number of function calls': ([], []),
#     'Runtime': ([], []),
#     'Steps': ([], [])
# }

# # For each row, add metrics to success/fail lists based on each run
# for _, row in df.iterrows():
#     for run in range(1, 6):
#         success = row[f'run{run}'] == 1
#         runtime = row[f'runtime {run}']
#         steps = row[f'steps {run}']
        
#         target = metrics['Number of lines'][0 if success else 1]
#         target.append(row['Number of lines'])
        
#         target = metrics['Number test cases'][0 if success else 1]
#         target.append(row['Number test cases'])
        
#         target = metrics['Number of function calls'][0 if success else 1]
#         target.append(row['Number of function calls'])
        
#         target = metrics['Runtime'][0 if success else 1]
#         target.append(runtime)
        
#         target = metrics['Steps'][0 if success else 1]
#         target.append(steps)


# # 1. Visualize the data
# # plt.figure(figsize=(10, 6))
# # sns.boxplot(x='successful', y='lines_of_code', data=data)
# # plt.title('Lines of Code by Success Status')
# # plt.xlabel('Successful (1) vs Unsuccessful (0)')
# # plt.ylabel('Lines of Code')
# # plt.show()

# # 2. Independent samples t-test
# successful_loc, unsuccessful_loc = metrics["Number of lines"]

# t_stat, p_value_ttest = stats.ttest_ind(successful_loc, unsuccessful_loc, equal_var=False)
# print(f"Independent t-test results:")
# print(f"t-statistic: {t_stat:.4f}")
# print(f"p-value: {p_value_ttest:.4f}")
# print(f"Mean lines of code (successful): {successful_loc.mean():.2f}")
# print(f"Mean lines of code (unsuccessful): {unsuccessful_loc.mean():.2f}")
# print()

# # 3. Logistic regression
# # model = logit('successful ~ lines_of_code', data=data).fit()
# # print("Logistic Regression Results:")
# # print(model.summary())

# # Get the p-value for the 'lines_of_code' coefficient
# # p_value_logit = model.pvalues['lines_of_code']
# # print(f"P-value for lines_of_code coefficient: {p_value_logit:.4f}")

# # 4. Mann-Whitney U test (non-parametric alternative if data is not normally distributed)
# u_stat, p_value_mw = stats.mannwhitneyu(successful_loc, unsuccessful_loc)
# print(f"\nMann-Whitney U test results:")
# print(f"U-statistic: {u_stat:.4f}")
# print(f"p-value: {p_value_mw:.4f}")

# 5. Plot the logistic regression model
# plt.figure(figsize=(10, 6))
# sns.regplot(x='lines_of_code', y='successful', data=data, logistic=True)
# plt.title('Logistic Regression: Probability of Success vs Lines of Code')
# plt.xlabel('Lines of Code')
# plt.ylabel('Probability of Success')
# plt.grid(True, alpha=0.3)
# plt.show()

# Interpretation
# alpha = 0.05
# print("\nInterpretation:")
# if p_value_ttest < alpha or p_value_mw < alpha:
#     print(f"There is a statistically significant relationship between lines of code and success status (p < {alpha}).")
# else:
#     print(f"There is no statistically significant relationship between lines of code and success status (p > {alpha}).")

# # Perform t-tests and print results
# print("T-test Results (Success vs Failure):")
# print("-" * 50)
# for metric, (success_vals, fail_vals) in metrics.items():
#     # Remove any NaN values
#     success_vals = np.array(success_vals)[~np.isnan(success_vals)]
#     fail_vals = np.array(fail_vals)[~np.isnan(fail_vals)]
    
#     if len(success_vals) > 0 and len(fail_vals) > 0:
#         t_stat, p_val = stats.ttest_ind(success_vals, fail_vals)
#         print(f"\n{metric}:")
#         print(f"Success mean: {np.mean(success_vals):.2f}")
#         print(f"Failure mean: {np.mean(fail_vals):.2f}")
#         print(f"t-statistic: {t_stat:.3f}")
#         print(f"p-value: {p_val:.3f}")
        
#         plt.figure(figsize=(8, 6))

#         sns.kdeplot(success_vals, fill=True, color='blue')
#         sns.kdeplot(fail_vals, fill=True, color='red')

#         plt.title(f'{metric} T-test\nt-stat={t_stat:.3f}, p={p_val:.3f}')
#         plt.xlabel(metric)
#         plt.ylabel('Frequency')
#         plt.legend()
        
#         # Save plot
#         plt.savefig(f'freq_{metric.lower().replace(" ", "_")}.png')
#         plt.close()


#         # Create density plot comparing success vs failure distributions
#         plt.figure(figsize=(8, 6))
#         success_density = stats.gaussian_kde(success_vals)
#         fail_density = stats.gaussian_kde(fail_vals)
        
#         x_range = np.linspace(min(min(success_vals), min(fail_vals)), 
#                              max(max(success_vals), max(fail_vals)), 
#                              200)
         
#         # plt.hist(success_vals, bins=10, color='green', label='Success')
#         # plt.hist(fail_vals, bins=10, color='red', label='Failure')
#         plt.plot(x_range, success_density(x_range), 'g-', label='Success', linewidth=2)
#         plt.plot(x_range, fail_density(x_range), 'r-', label='Failure', linewidth=2)
        
#         plt.title(f'{metric} Distribution\nt-stat={t_stat:.3f}, p={p_val:.3f}')
#         plt.xlabel(metric)
#         plt.ylabel('Density')
#         plt.legend()
        
#         # Save density plot
#         plt.savefig(f'density_{metric.lower().replace(" ", "_")}.png')
#         plt.close()

#         # Create violin plot comparing success vs failure distributions
#         plt.figure(figsize=(8, 6))
#         plt.violinplot([success_vals, fail_vals], positions=[1, 2])
#         plt.xticks([1, 2], ['Success', 'Failure'])
#         plt.title(f'{metric} Distribution\nt-stat={t_stat:.3f}, p={p_val:.3f}')
#         plt.ylabel(metric)
        
#         # Add means as points
#         plt.plot([1], [np.mean(success_vals)], 'ko', label='Mean')
#         plt.plot([2], [np.mean(fail_vals)], 'ko')
        
#         # Save plot
#         plt.savefig(f't_test_{metric.lower().replace(" ", "_")}.png')
#         plt.close()
