. /work/gyeongwk/anaconda3/etc/profile.d/conda.sh

# Map paper_id to env_name
paper_id=$2
if [ "$paper_id" = "2110.03485" ]; then
    conda_env="cartoonx"
elif [ "$paper_id" = "2205.00048" ]; then
    conda_env="jme"
elif [ "$paper_id" = "2303.11932" ]; then
    conda_env="fact"
elif [ "$paper_id" = "2309.05569" ]; then
    conda_env="iti-gen"
else
    conda_env="None"
fi


echo "Activating $conda_env"
conda activate $conda_env

python test_func_dependency.py "$@"
