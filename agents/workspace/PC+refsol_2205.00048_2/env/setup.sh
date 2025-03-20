# Create environment
conda env create -f env/jme.yml
conda activate jme
# Download dataset Librarything
python3 src/BERT4Rec/download_pretrained.py

python << END
print('------- JME-Fairness Reproducibility Study -------\n')
print('Hello, welcome to the JME Reproducability Study code repository\n\n\
To plot all figures in our reproducibility study:\n   \
conda activate jme\n
bash experiments/run_files/plot_reproduction_results.sh \n')
END