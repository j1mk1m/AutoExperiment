# Paper Name 
Neurips 2025 Submission

### Downloading the Dataset
[Huggingface Link]()

Datapoint

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
- `--retrieval agent`: options are "no" and "agent"
- `--code-retrieval agent`: options are "no", "full", "ast", and "embedding"


### Experiments
These are instructions on how to reproduce the experiments in the paper.

#### Verifier
Run the verifier with the following command

#### Fixed vs Dynamic
We reproduce the "agentless" harness.