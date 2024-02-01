# Copy to writeable directory
cp -r /app/tmp /app/MLAgentBench
cd /app/MLAgentBench

conda env create -n myenv -f environment.yml

conda run --no-capture-output -n myenv python -u MLAgentBench/runner.py --task task --log-dir logs --work-dir workspace \
	--max-steps 20 \
	--max-time 72000 \
	--llm-name gpt-4-1106-preview \
	--edit-script-llm-name gpt-4-1106-preview \
	--fast-llm-name gpt-3.5-turbo-1106

