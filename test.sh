
### Test utils ###
#python utils/find_conda_env.py --combined_id 0000.00000_0 --file experiments-light.csv  # micn_env
#python utils/find_conda_yml.py --combined_id 0000.00000_0 --file experiments-light.csv  

agent='BasicPromptAgent' # MLAgentBench, refsol, AutoAgent, BasicPromptAgent
combined_id='2205.00048_2'
model='gpt-4o-mini'
mode='PC+refsol'
retrieval='agent'

## Full Run $$
bash run_exp_from_env.sh --_tags test --agent $agent --split MLRC --mode $mode --retrieval $retrieval --combined_id $combined_id --model $model --verbose
