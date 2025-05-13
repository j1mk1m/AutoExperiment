# Test script
combined_id="2205.00048_0" # 2110.03485, 2205.00048, 2303.11932, 2309.05569
agent='ReAct' 
environment="MLAgentBench"
memory="Full"
model='o1'
retrieval='agent'
code_retrieval='agent'

## Full Run $$
bash run_exp_from_env.sh --_tags test --combined-id $combined_id --agent $agent --environment $environment --memory $memory --model-engine $model --retrieval $retrieval --code-retrieval $code_retrieval --verbose
