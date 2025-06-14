# Test script
combined_id="2205.00048_0" # 2110.03485, 2205.00048, 2303.11932, 2309.05569
agent='ReAct' 
memory="Full"
model='gpt-4o-mini'
retrieval='agent'
code_retrieval='agent'

## Full Run $$
bash run_exp_from_env.sh --_tags test --combined-id $combined_id --agent $agent --memory $memory --model-engine $model --retrieval $retrieval --code-retrieval $code_retrieval --verbose
