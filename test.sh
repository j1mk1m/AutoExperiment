
combined_id='2205.00048_2'
agent='ReAct' 
environment="MLAgentBench"
memory="Full"
model='gpt-4o-mini'

## Full Run $$
singularity exec -B /work/gyeongwk --nv ../experiment_tools/base_img.sif bash run_exp_from_env.sh --_tags test --combined_id $combined_id --agent $agent --environment $environment --memory $memory --model $model --verbose
