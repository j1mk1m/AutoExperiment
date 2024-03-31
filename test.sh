
### Test utils ###
#python utils/find_conda_env.py --combined_id 0000.00000_0 --file experiments-light.csv  # micn_env
#python utils/find_conda_yml.py --combined_id 0000.00000_0 --file experiments-light.csv  

agent='AutoAgent' # MLAgentBench, refsol, AutoAgent
file='experiments-light.csv'
combined_id='0000.00000_0'
model='gpt-3.5-turbo-0125' # gpt-4-1106-preview
## Full Run $$
bash run_exp_from_env.sh --_tags test --agent $agent --mode FC --file $file --combined_id $combined_id --model $model 

## How to run runner.py ##
# python runner.py --_tags test --agent refsol --mode FC --file experiments-light.csv --combined_id 0000.00000_0
# python runner.py --_tags test --agent MLAgentBench --mode FC --file experiments-light.csv --combined_id 0000.00000_0
