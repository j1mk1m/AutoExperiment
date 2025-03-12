. /work/gyeongwk/anaconda3/etc/profile.d/conda.sh

# Map paper_id to env_name
conda_env=$(python utils/find_conda_env.py "$@")

echo "Activating $conda_env"
conda activate $conda_env
# export TMPDIR=/work/gyeongwk/pip_cache
# pip install -r requirements.txt --cache-dir /work/gyeongwk/pip_cache

python runner.py "$@"
