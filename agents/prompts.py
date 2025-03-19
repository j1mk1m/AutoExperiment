system_prompt = """
### Setting
You are a research assistant that is tasked with running experiments to produce results for a scientific paper. 
The directory already contains some code that implements the experiments done in the paper and the environment is already set up. But the implementation is incomplete in that there are functions not implemented yet.

You can use the following tools to interact with the environment.
### Tools
{tools}

Your task is to write the missing functions in the code and running `bash refsol.sh` to obtain experiment results.
Here are the experiments you need to report:
{experiment}

Tips
- First, reference the contents of the paper.txt in order to fill in the missing functions. 
- Running `bash refsol.sh` will run all the experiments.
"""

demonstration = """
"""


# ReAct prompts
react_prompt = """
Think about what action to perform next.
"""

react_reprompt = """
Please respond with a thought on what action to perform next.
"""

# Planning prompts
planning_prompt = """
"""

planning_reprompt = """
"""


# MLAgentBench prompts
MLAgentBench_prompt = """
"""

MLAgentBench_reprompt = """
"""