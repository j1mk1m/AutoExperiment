# Requirements 
- PyTorch
- docker https://docs.docker.com/engine/install/ubuntu/#install-from-a-package
- nvidia-container-toolkit (for gpu use) https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#installing-with-apt 

# How to run
```
python main.py --baseline MLAgentBench --mode FC
```
Supported Baselines
- MLAgentBench (default)

Modes
- FC (full code / default)
- NC (no code)

# Create new datapoints
```
cd dataset
mkdir {paper_id}
cd {paper_id}
git clone {repo_url}
# rename repo directory to code/
touch environment.yml # figure out necessary dependencies
touch paper.txt # Copy contents of paper (see below)
# Add row(s) to experiments.csv (paper_id, exp_id, exp_detail, exp_result)
```
## Paper to txt pipeline
1. Find paper on ArXiv
2. Change url from arxiv -> ar5iv
3. Download webpage (html)
4. ``` pandoc -o paper.txt {webpage.html} ``` https://pandoc.org/MANUAL.html
5. Delete experiment results in paper.txt
