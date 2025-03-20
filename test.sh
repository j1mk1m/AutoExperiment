# Test script
combined_id='2110.03485_' # 2110.03485, 2205.00048, 2303.19932, 2309.05569
agent='refsol' 
environment="MLAgentBench"
memory="SlidingWindow"
model='gpt-4o-mini'

## Full Run $$
singularity exec -B /work/gyeongwk --nv ../experiment_tools/base_img.sif bash run_exp_from_env.sh --_tags test --combined-id $combined_id --agent $agent --environment $environment --memory $memory --model-engine $model --verbose
