. /work/gyeongwk/anaconda3/etc/profile.d/conda.sh

yml_path=$(python utils/find_conda_yml.py "$@")
echo "Creating conda env from yml in $yml_path"
conda env create -n myenv --file=$yml_path
conda activate myenv
pip install -r requirements.txt --cache-dir /work/gyeongwk/pip_cache

python runner.py "$@"
