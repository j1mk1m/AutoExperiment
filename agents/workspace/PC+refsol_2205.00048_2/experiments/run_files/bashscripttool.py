# Write experiment GPU time to file
models = ['BPRMF', 'LDA', 'PureSVD', 'SLIM', 'WRMF', 'CHI2', 'HT', 'KLD', 'LMWI', 'LMWU', 'SVD', 'NNI', 'NNU', 'PLSA', 'POP', 'Random', 'RM1', 'RM2', 'RSV', 'RW', 'UIR']
dataset = 'ml-1m'

with open("experiments/run_files/run_metrics_movielens.sh", "w") as f:
    for i in range(len(models)):
        f.write('mkdir src/outputs/ml-1m/Experiment_' + str(i+1) +'_' + models[i] +'\n')
        f.write('mkdir src/outputs/ml-1m/Experiment_' + str(i+1) +'_' + models[i] +'/Gender\n')
        f.write('mkdir src/outputs/ml-1m/Experiment_' + str(i+1) +'_' + models[i] +'/Age\n')
        f.write('python3 src/run_metric.py --data ml-1m --model '+ models[i]+' --conduct sh --age N\n')
        f.write('python3 src/run_metric.py --data ml-1m --model '+ models[i]+' --conduct st --age N\n')
        f.write('mv src/outputs/ml-1m/*.json src/outputs/ml-1m/Experiment_' + str(i+1) +'_' + models[i] +'/Gender\n')
        f.write('python3 src/run_metric.py --data ml-1m --model '+ models[i] +' --conduct sh --age Y\n')
        f.write('python3 src/run_metric.py --data ml-1m --model '+ models[i]+' --conduct st --age Y\n')
        f.write('mv src/outputs/ml-1m/*.json src/outputs/ml-1m/Experiment_' + str(i+1) +'_' + models[i] +'/Age\n\n')

with open("experiments/run_files/run_metrics_librarything.sh", "w") as f:
    for i in range(len(models)):
        f.write('mkdir src/outputs/lt/Experiment_' + str(i+1) +'_' + models[i] +'\n')
        f.write('python3 src/run_metric.py --data lt --model '+ models[i]+' --conduct sh --ndatapoints 15000\n')
        f.write('python3 src/run_metric.py --data lt --model '+ models[i]+' --conduct st --ndatapoints 15000\n')
        f.write('mv src/outputs/lt/*.json src/outputs/lt/Experiment_' + str(i+1) +'_' + models[i] +'\n')


with open("experiments/run_files/plot_experiments_detrank.sh", 'w') as f:
    for i in range(len(models)):
        f.write('python3 src/utils/postprocessing.py --fig2_models '+ models[i]+' --figures 2 --group lt\n')
        f.write('python3 src/utils/postprocessing.py --fig2_models '+ models[i]+' --figures 2 --group Age\n')
        f.write('python3 src/utils/postprocessing.py --fig2_models '+ models[i]+' --figures 2 --group Gender\n')
