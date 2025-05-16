# From Reproduction to Replication: Evaluating Research Agents with Progressive Code Masking
Gyeongwon James Kim, Alex Wilf, Louis-Philippe Morency, Daniel Fried (Carnegie Mellon University)

### Dataset
Access the dataset via [Huggingface](https://huggingface.co/datasets/j1mk1m/AutoExperiment) r run 
```
cd dataset
python process_dataset.py
```

For each repository, it is recommended to create conda environments from the provided yml files. Refer to [conda-docs](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) on how to manage conda environments.

Note: Each sample in the dataset is identified with the "combined_id" ({paper_id}_{function_id}). For example, 2205.00048_0 represents a sample where code repository from paper 2205.00048 is used and function 0 is masked out. Also, 2205.00048_0,1 represents a sample with functions 0 and 1 masked out.


### Running the Agent
Basic example of running the agent.
```
bash run_exp_from_env.sh \
    --combined_id 2205.00048_0 \
    --agent ReAct \
    --memory Full \
    --model-engine gpt-4o
```

Optional arguments

- `--max-agent-steps 50`: maximum number of action-taking steps allowed
- `--compute-budget 1.0`: compute budget allowed in dollars
- `--max-compute-time 1800`: maximum time (in seconds) allowed for the agent
- `--retrieval full`: options are "no", "full", "embedding", and "oracle"
- `--code-retrieval full`: options are "no", "full", "ast", and "embedding"


### Experiments
These are instructions on how to reproduce the experiments in the paper.

#### Removing $n \ge 1$ functions (Section 3.1)
Preparing the dataset will generate files `mlrc_n_[1,2,3,4,5].jsonl`. Each sample can be run individually with its combined_id. A way to run a batch of samples is described in section [Batched run jobs using wandb sweeps](#batched-run-jobs-using-wandb-sweeps).

#### The Pass@k Gap (Section 3.2)
After running $k$ passes through the same set of datapoints, collect all logs in one directory. Then, run 
```
cd agents
python log_parser.py --log-dir path/to/log/directory
```
This will generate a csv file in the log directory containing the generated Python code for each run. Then, run
```
python verifier.py --log-dir path/to/log/directory --model-engine model_name
```
This will run the model verifier with the specified LLM engine and output a file containing verifier selections.

#### Scaling Interactivity and Test-Time Compute in Agents (Section 3.3)
We reproduce the "agentless" harness. This can be run by setting `--agent Agentless` and disregarding any parameter related to agent architecture. The maximum number of reasoning tokens can be controlled with `--max-reasoning-tokens` paramter.

### Dependence on Natural Language (Section 3.4)
No vs Full settings are set using `--retrieval [full, no]` parameter. By default "full" setting is used. In the "Full" setting, agents are given contents of the research paper (paper.txt) as part of the repository. In the "No" setting, this is not given.

### Agent Architectures and Backbones (Section 3.5)
Any combination of agent architecture and model backbone can be tested.

- `--agent`: one of [ReAct, Planning, MLAgentBench]. These are prompting techniques described in the paper.
- `--memory`: one of [Full, SlidingWindow, Summary]. Note, SlidingWindow and Summary history management strategies take in an additional parameter `--lookback k` as described in the paper.
- `--model-engine`: any LLM can be used via litellm library. The ones used in the paper are [gpt-4o, gpt-4o-mini, anthropic/claude-3-5-sonnet-20240620, anthropic/claude-3-7-sonnet-20250219, o1, o3-mini].


### Batched run jobs using wandb sweeps
We make use of wandb sweeps to organize batched runs of multiple samples. Use the template yaml files in `deploy` directory and run
```
python deploy_sweeps.py example.yml
```
This creates a wandb sweep. More information [here](https://docs.wandb.ai/guides/sweeps/).