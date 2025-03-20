echo Experiment 1
python src/run_metric.py --ndatapoints 1 --conduct sh --model Pop --age N
echo Experiment 2
python src/run_metric.py --ndatapoints 1 --conduct sh --model Pop --age Y
echo Experiment 3
python src/run_metric.py --ndatapoints 1 --model BPRMF --age N
echo Experiment 4
python src/run_metric.py --ndatapoints 1 --conduct sh --model BPRMF --age Y
