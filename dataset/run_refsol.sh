# Copy to writeable directory
cp -r /app/tmp /app/env
cd /app/env

# Create environment and install dependencies
conda env create -n myenv -f environment.yml
source /home/user/micromamba/bin/activate myenv

bash refsol.sh
