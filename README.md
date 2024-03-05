# Requirements 
- docker https://docs.docker.com/engine/install/ubuntu/#install-from-a-package
- nvidia-container-toolkit (for gpu use) https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#installing-with-apt 

# How to run
First, set up base docker image:
```
cd baselines
docker build -t base_image .
```
Then, run the following:
```
python main.py --baseline MLAgentBench --mode FC --local
```
### Arguments
- `exp_file`: experiment csv file to be used. Put custom csv files inside `dataset/experiment_csvs/` directory
  - `experiments-light.csv` (default)
- `baseline`: pick baseline/agent to use
  - MLAgentBench (default)
- `mode`
  - FC (full code/ default)
  - NC (no code)
- `local`
  - if flag is set, baseline will run locally and not on docker
  - if flag is not set (default), baseline will run on docker
 
# Repository Structure
`dataset/`: contains repositories in the dataset, DataLoader class, and experiment csvs
- `0000.00000/`: organized by paper id (usually ArXiv id)
  - `code/`: contains cloned respository
  - `paper.txt`: contents of paper in txt format with experiment results removed
  - `environment.yml`: environment yaml file used to reproduce the environment in the paper
- `experiment_csvs/`: contains csv files of experiments
- `refsols`: contains refsol files of experiments (if applicable)
- `dataset.py`: contains DataLoader class that creates workspace for each data point given as argument a experiment csv file
- `test_refsol.py`: if you have refsols for experiments, run this to verify that it works inside docker

`workspace/`: this directory will be populated by the DataLoader in `dataset/dataset.py` and used as cache. To save space, use the `remove_workspace` function in `dataset/dataset.py` to delete cached workspaces. This directory is organized by `mode/paper_id/exp_id` (e.g. `FC/0000.00000/0`)

`baselines`: contains implementation of baselines used
- `MLAgentBench`: contains code implementing MLAgentBench baseline
- `openai_api_key.txt`: put your OpenAI api key in this file
- `Dockerfile`: use this to create the base docker image that will be used for all data points
- `run.sh`: bash script used to run the baseline inside docker
- `tmp/`: temporary directory that will store log and output of a run
- `run_baseline.py`: python script that sets up baseline and runs for one data point (experiment)

`main.py`: main script to run Agents over experiments. See How to Run section above.

# Create new datapoints
Here are instructions on how to expand the dataset.
```
cd dataset
mkdir {paper_id}
cd {paper_id}
git clone {repo_url}
# rename repo directory to code/
touch environment.yml # figure out necessary dependencies
touch paper.txt # Copy contents of paper (see below)
# Add row(s) to experiments.csv (paper_id, exp_id, description, result, ref_sol(optional))
# See experiments-light.csv for example format
```
## Paper to txt pipeline
1. Find paper on ArXiv
2. Change url from arxiv -> ar5iv
3. Download webpage (html)
4. ``` pandoc -o paper.txt {webpage.html} ``` https://pandoc.org/MANUAL.html
5. Delete experiment results in paper.txt
