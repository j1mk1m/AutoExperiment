# Tasks
General Task: Given information about a scientific experiment with (potentially) incomplete code, reproduce the experiment and output a numerical result.
### Settings
- **Full Code**: full repository given as input. Goal is to reproduce the experiment by running a sequence of commands
- **Partial Code**: one Python function will be blocked and replaced with a GPT-4 generated docstring. The goal is to fill in the missing function, then run the sequence of commands to reproduce the experiment
- **Partial Code + Refsol** : one Python function will be blocked and replaced with a GPT-generated docstring. In addition, we give the refsol containing sequence of commands to run. The goal is to fill in the missing function so that running the refsol runs the desired experiment

| Setting                           | Code               | paper contents | experiment description | refsol |
| --------------------------------- | ------------------ | -------------- | ---------------------- | ------ |
| Full Code (FC)                    | Full               | Yes            | Yes                    | No     |
| Partial Code (PC)                 | 1 function removed | Yes            | Yes                    | No     |
| Partial Code + Refsol (PC+refsol) | 1 function removed | Yes            | Yes                    | Yes    |

# Dataset
### Dataset Creation
This explains the process of creating new data points for the benchmark. 
1. First, pick a research paper and find the repository for its code.
2. Create a directory named with the paper id in `dataset/{paper_id}`
#### Paper Processing
1. Find paper on arxiv
2. Change URL to ar5iv and download the HTML5
3. Use [Pandoc](https://pandoc.org/MANUAL.html)to convert html to .txt
	1. `pandoc -o paper.txt paper.html`
4. Extract specific experiments and results from the paper
#### Code Processing
1. Git clone repository inside `dataset/{paper_id}` directory as `code/`
2. Set up a conda environment and verify that it has all necessary dependencies
3. Create a `environment.yml` file from the conda environment
4. Identify main Python functions to remove in the PC setting and list them in `functions.json`
	1. Each function should contain `script` (relative path to Python file), `name` (name of Python function), `line_start`, `line_end` (location of Python function in file), `description` (GPT-4 generated docstring)
	2. Helper script to generate the docstrings is in `dataset/util.py`
### Dataset Directory Structure
- `dataset/`
	- `{paper_id}/` :contain contents for one paper
		- `code/` : contains full repository implementing experiments in paper
		- `environment.yml` : yml file with necessary dependencies 
		- `paper.txt` : contents of paper in txt
		- `functions.json`: contains list of main Python functions to remove for the Partial Code setting
	- `experiment_csvs/` : contains csv files for experiments
		- `experiments-light.csv` : each csv file should have rows paper_id, exp_id, description, result, refsol, environment
	- `dataset.py` : contains helper functions to prepare workspace for specific datapoint `{paper_id}_{exp_id}`
### Experiment CSV files
- paper_id: arxiv id of the paper
- exp_id: numbering of the experiments for one paper
- combined_id (optional): `{paper_id}_{exp_id}`
- description: text description of the experiment including all parameters and what to return
- result: expected result of the experiment. This should be a single numerical value
- refsol: reference sequence of commands to run to reproduce the experiment. This should be able to be copied into a bash file and run.
- environment: name of conda environment if already exists

### Datapoint
One datapoint is defined by `{paper_id}_{exp_id}` (one specific experiment from a paper).
We set up input, output pair as follows:

Input `X`:
- `path`: path to workspace directory created by `dataset.py`
	- `paper.txt`: paper contents
	- `experiment.txt`: experiment description in txt form
	- `code/` : repository (FC: full repo, PC: one Python function removed)
	- `refsol.sh` : bash file containing refsol commands (PC+refsol setting)
- `mode`: (meta) FC, PC, PC+refsol, etc
- `paper_id` (meta)
- `exp_id` (meta)
- `environment` (meta)
- `description`: string description of the experiment
- `func_to_block`: dictionary containing information about removed function (PC setting)
	- `script`: relative path to Python script with the missing function
	- `name`: name of missing function
	- `description`: GPT-4 generated docstring for the function
	- `line_start`, `line_end`: location of missing function in file
Output `y`:
- `result`: single numerical measurement after running given experiment

Sample Data point
![sample_datapoint](https://github.com/j1mk1m/AutoExperiment/assets/68579388/4a26384e-6bd7-4bc3-9f48-801852b44fd1)

Sample function removed
![AutoExperiment_sample_func](https://github.com/j1mk1m/AutoExperiment/assets/68579388/c84a5f23-fdff-4577-9e15-8f42859443d0)

# Testing Framework
![AutoExperiment Pipeline](https://github.com/j1mk1m/AutoExperiment/assets/68579388/0910eeef-b6fe-4125-813f-a13fbcd6b23a)

# Agents
![AutoExperiment Agent Architecture](https://github.com/j1mk1m/AutoExperiment/assets/68579388/db1bd038-8f44-473a-8830-f5eb26637e3b)

We characterize an Agent with the following properties:
- Architecture
	- Prompting: what kind of prompting strategies are used
	- Memory: how does the agent maintain memory
	- Tool call: what are the tool calls available
- LLM model: which models does the agent use (e.g. GPT-3.5, GPT-4, etc)
	- Instruction fine-tuned models
	- Code Generation tuned models
	- Function calling models

Baseline Agent architectures
- `BasicPromptAgent`: this agent prompts LLM one time to generate missing code and to generate the sequence of commands to run
- `MLAgentBench`: this agent prompts the LLM to generate a research plan, 
- `AutoAgent`: this agent uses similar tool calls and prompting strategies as MLAgentBench but uses function calling, prompting by parts, and better memory management

