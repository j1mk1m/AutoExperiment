# Copy to writeable directory
cp -r /app/tmp /app/MLAgentBench
cd /app/MLAgentBench

# Create environment and install dependencies
conda env create -n myenv -f environment.yml
source /home/user/micromamba/bin/activate myenv
pip install -r requirements.txt

# Run MLAgentBench
python -u MLAgentBench/runner.py --task task --log-dir logs --work-dir workspace \
	--max-steps 200 \
	--max-time 72000 \
	--llm-name gpt-4-1106-preview \
	--edit-script-llm-name gpt-4-1106-preview \
	--fast-llm-name gpt-3.5-turbo-1106

