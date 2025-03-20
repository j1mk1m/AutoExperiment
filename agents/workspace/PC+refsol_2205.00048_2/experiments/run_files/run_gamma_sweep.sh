# Gamma 0.0
mkdir src/outputs/ml-1m/parameter_sweep/gamma/00
mkdir src/outputs/ml-1m/parameter_sweep/gamma/00/Gender
mkdir src/outputs/ml-1m/parameter_sweep/gamma/00/Age

python3 src/run_metric.py --data ml-1m --model BPRMF --conduct sh --age N --gamma 0.01
python3 src/run_metric.py --data ml-1m --model BPRMF --conduct st --age N --gamma 0.01
mv src/outputs/ml-1m/*.json src/outputs/ml-1m/parameter_sweep/gamma/00/Gender

python3 src/run_metric.py --data ml-1m --model BPRMF --conduct sh --age Y --gamma 0.01
python3 src/run_metric.py --data ml-1m --model BPRMF --conduct st --age Y --gamma 0.01
mv src/outputs/ml-1m/*.json src/outputs/ml-1m/parameter_sweep/gamma/00/Age
#
# # Gamma 0.2
# mkdir src/outputs/ml-1m/parameter_sweep/gamma/02
# mkdir src/outputs/ml-1m/parameter_sweep/gamma/02/Gender
# mkdir src/outputs/ml-1m/parameter_sweep/gamma/02/Age
#
# python3 src/run_metric.py --data ml-1m --model BPRMF --conduct sh --age N --gamma 0.2
# python3 src/run_metric.py --data ml-1m --model BPRMF --conduct st --age N --gamma 0.2
# mv src/outputs/ml-1m/*.json src/outputs/ml-1m/parameter_sweep/gamma/02/Gender
#
# python3 src/run_metric.py --data ml-1m --model BPRMF --conduct sh --age Y --gamma 0.2
# python3 src/run_metric.py --data ml-1m --model BPRMF --conduct st --age Y --gamma 0.2
# mv src/outputs/ml-1m/*.json src/outputs/ml-1m/parameter_sweep/gamma/02/Age
#
# # Gamma 0.4
# mkdir src/outputs/ml-1m/parameter_sweep/gamma/04
# mkdir src/outputs/ml-1m/parameter_sweep/gamma/04/Gender
# mkdir src/outputs/ml-1m/parameter_sweep/gamma/04/Age
#
# python3 src/run_metric.py --data ml-1m --model BPRMF --conduct sh --age N --gamma 0.4
# python3 src/run_metric.py --data ml-1m --model BPRMF --conduct st --age N --gamma 0.4
# mv src/outputs/ml-1m/*.json src/outputs/ml-1m/parameter_sweep/gamma/04/Gender
#
# python3 src/run_metric.py --data ml-1m --model BPRMF --conduct sh --age Y --gamma 0.4
# python3 src/run_metric.py --data ml-1m --model BPRMF --conduct st --age Y --gamma 0.4
# mv src/outputs/ml-1m/*.json src/outputs/ml-1m/parameter_sweep/gamma/04/Age
#
# # Gamma 0.6
# mkdir src/outputs/ml-1m/parameter_sweep/gamma/06
# mkdir src/outputs/ml-1m/parameter_sweep/gamma/06/Gender
# mkdir src/outputs/ml-1m/parameter_sweep/gamma/06/Age
#
# python3 src/run_metric.py --data ml-1m --model BPRMF --conduct sh --age N --gamma 0.6
# python3 src/run_metric.py --data ml-1m --model BPRMF --conduct st --age N --gamma 0.6
# mv src/outputs/ml-1m/*.json src/outputs/ml-1m/parameter_sweep/gamma/06/Gender
#
# python3 src/run_metric.py --data ml-1m --model BPRMF --conduct sh --age Y --gamma 0.6
# python3 src/run_metric.py --data ml-1m --model BPRMF --conduct st --age Y --gamma 0.6
# mv src/outputs/ml-1m/*.json src/outputs/ml-1m/parameter_sweep/gamma/06/Age
#
# # Gamma 0.8
# mkdir src/outputs/ml-1m/parameter_sweep/gamma/08
# mkdir src/outputs/ml-1m/parameter_sweep/gamma/08/Gender
# mkdir src/outputs/ml-1m/parameter_sweep/gamma/08/Age
#
# python3 src/run_metric.py --data ml-1m --model BPRMF --conduct sh --age N --gamma 0.8
# python3 src/run_metric.py --data ml-1m --model BPRMF --conduct st --age N --gamma 0.8
# mv src/outputs/ml-1m/*.json src/outputs/ml-1m/parameter_sweep/gamma/08/Gender
#
# python3 src/run_metric.py --data ml-1m --model BPRMF --conduct sh --age Y --gamma 0.8
# python3 src/run_metric.py --data ml-1m --model BPRMF --conduct st --age Y --gamma 0.8
# mv src/outputs/ml-1m/*.json src/outputs/ml-1m/parameter_sweep/gamma/08/Age

# Gamma 0.9
# mkdir src/outputs/ml-1m/parameter_sweep/gamma/09
# mkdir src/outputs/ml-1m/parameter_sweep/gamma/09/Gender
# mkdir src/outputs/ml-1m/parameter_sweep/gamma/09/Age

# python3 src/run_metric.py --data ml-1m --model BPRMF --conduct sh --age N --gamma 0.9
# python3 src/run_metric.py --data ml-1m --model BPRMF --conduct st --age N --gamma 0.9
# mv src/outputs/ml-1m/*.json src/outputs/ml-1m/parameter_sweep/gamma/09/Gender

# python3 src/run_metric.py --data ml-1m --model BPRMF --conduct sh --age Y --gamma 0.9
# python3 src/run_metric.py --data ml-1m --model BPRMF --conduct st --age Y --gamma 0.9
# mv src/outputs/ml-1m/*.json src/outputs/ml-1m/parameter_sweep/gamma/09/Age
