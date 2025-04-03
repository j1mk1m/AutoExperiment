# Test script
combined_id='2110.03485_0' # 2110.03485, 2205.00048, 2303.19932, 2309.05569
agent='ReAct' 
environment="MLAgentBench"
memory="SlidingWindow"
model='gpt-4o-mini'

## Full Run $$
bash run_exp_from_env.sh --_tags test --combined-id $combined_id --agent $agent --environment $environment --memory $memory --model-engine $model --verbose
