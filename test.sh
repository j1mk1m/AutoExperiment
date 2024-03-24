
### Test utils ###
#python utils/find_conda_env.py --combined_id 0000.00000_0 --file experiments-light.csv  # micn_env
#python utils/find_conda_yml.py --combined_id 0000.00000_0 --file experiments-light.csv  

## Full Run $$
bash run_exp_from_env.sh --_tags test --agent MLAgentBench --mode FC --file experiments-light.csv --combined_id 0000.00000_0 --model gpt-4-1106-preview
# bash run_exp_from_env.sh --_tags test --agent MLAgentBench --mode FC --file experiments-light.csv --combined_id 0000.00000_0 --model gpt-3.5-turbo-1106
#bash run_exp_from_env.sh --_tags test --agent refsol --mode FC --file experiments-light.csv --combined_id 0000.00000_0

## How to run runner.py ##
# python runner.py --_tags test --agent refsol --mode FC --file experiments-light.csv --combined_id 0000.00000_0
# python runner.py --_tags test --agent MLAgentBench --mode FC --file experiments-light.csv --combined_id 0000.00000_0
