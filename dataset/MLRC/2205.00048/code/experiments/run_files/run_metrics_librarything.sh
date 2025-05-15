mkdir src/outputs/lt/Experiment_1_BPRMF
#python3 src/run_metric.py --data lt --model BPRMF --conduct sh --ndatapoints 15000
#python3 src/run_metric.py --data lt --model BPRMF --conduct st --ndatapoints 15000
mv src/outputs/lt/*.json src/outputs/lt/Experiment_1_BPRMF
mkdir src/outputs/lt/Experiment_2_LDA
python3 src/run_metric.py --data lt --model LDA --conduct sh --ndatapoints 15000
python3 src/run_metric.py --data lt --model LDA --conduct st --ndatapoints 15000
mv src/outputs/lt/*.json src/outputs/lt/Experiment_2_LDA
mkdir src/outputs/lt/Experiment_3_PureSVD
python3 src/run_metric.py --data lt --model PureSVD --conduct sh --ndatapoints 15000
python3 src/run_metric.py --data lt --model PureSVD --conduct st --ndatapoints 15000
mv src/outputs/lt/*.json src/outputs/lt/Experiment_3_PureSVD
mkdir src/outputs/lt/Experiment_4_SLIM
python3 src/run_metric.py --data lt --model SLIM --conduct sh --ndatapoints 15000
python3 src/run_metric.py --data lt --model SLIM --conduct st --ndatapoints 15000
mv src/outputs/lt/*.json src/outputs/lt/Experiment_4_SLIM
mkdir src/outputs/lt/Experiment_5_WRMF
python3 src/run_metric.py --data lt --model WRMF --conduct sh --ndatapoints 15000
python3 src/run_metric.py --data lt --model WRMF --conduct st --ndatapoints 15000
mv src/outputs/lt/*.json src/outputs/lt/Experiment_5_WRMF
mkdir src/outputs/lt/Experiment_6_CHI2
python3 src/run_metric.py --data lt --model CHI2 --conduct sh --ndatapoints 15000
python3 src/run_metric.py --data lt --model CHI2 --conduct st --ndatapoints 15000
mv src/outputs/lt/*.json src/outputs/lt/Experiment_6_CHI2
mkdir src/outputs/lt/Experiment_7_HT
python3 src/run_metric.py --data lt --model HT --conduct sh --ndatapoints 15000
python3 src/run_metric.py --data lt --model HT --conduct st --ndatapoints 15000
mv src/outputs/lt/*.json src/outputs/lt/Experiment_7_HT
mkdir src/outputs/lt/Experiment_8_KLD
python3 src/run_metric.py --data lt --model KLD --conduct sh --ndatapoints 15000
python3 src/run_metric.py --data lt --model KLD --conduct st --ndatapoints 15000
mv src/outputs/lt/*.json src/outputs/lt/Experiment_8_KLD
mkdir src/outputs/lt/Experiment_9_LMWI
python3 src/run_metric.py --data lt --model LMWI --conduct sh --ndatapoints 15000
python3 src/run_metric.py --data lt --model LMWI --conduct st --ndatapoints 15000
mv src/outputs/lt/*.json src/outputs/lt/Experiment_9_LMWI
mkdir src/outputs/lt/Experiment_10_LMWU
python3 src/run_metric.py --data lt --model LMWU --conduct sh --ndatapoints 15000
python3 src/run_metric.py --data lt --model LMWU --conduct st --ndatapoints 15000
mv src/outputs/lt/*.json src/outputs/lt/Experiment_10_LMWU
mkdir src/outputs/lt/Experiment_11_SVD
python3 src/run_metric.py --data lt --model SVD --conduct sh --ndatapoints 15000
python3 src/run_metric.py --data lt --model SVD --conduct st --ndatapoints 15000
mv src/outputs/lt/*.json src/outputs/lt/Experiment_11_SVD
mkdir src/outputs/lt/Experiment_12_NNI
python3 src/run_metric.py --data lt --model NNI --conduct sh --ndatapoints 15000
python3 src/run_metric.py --data lt --model NNI --conduct st --ndatapoints 15000
mv src/outputs/lt/*.json src/outputs/lt/Experiment_12_NNI
mkdir src/outputs/lt/Experiment_13_NNU
python3 src/run_metric.py --data lt --model NNU --conduct sh --ndatapoints 15000
python3 src/run_metric.py --data lt --model NNU --conduct st --ndatapoints 15000
mv src/outputs/lt/*.json src/outputs/lt/Experiment_13_NNU
mkdir src/outputs/lt/Experiment_14_PLSA
python3 src/run_metric.py --data lt --model PLSA --conduct sh --ndatapoints 15000
python3 src/run_metric.py --data lt --model PLSA --conduct st --ndatapoints 15000
mv src/outputs/lt/*.json src/outputs/lt/Experiment_14_PLSA
mkdir src/outputs/lt/Experiment_15_POP
python3 src/run_metric.py --data lt --model POP --conduct sh --ndatapoints 15000
python3 src/run_metric.py --data lt --model POP --conduct st --ndatapoints 15000
mv src/outputs/lt/*.json src/outputs/lt/Experiment_15_POP
mkdir src/outputs/lt/Experiment_16_Random
python3 src/run_metric.py --data lt --model Random --conduct sh --ndatapoints 15000
python3 src/run_metric.py --data lt --model Random --conduct st --ndatapoints 15000
mv src/outputs/lt/*.json src/outputs/lt/Experiment_16_Random
mkdir src/outputs/lt/Experiment_17_RM1
python3 src/run_metric.py --data lt --model RM1 --conduct sh --ndatapoints 15000
python3 src/run_metric.py --data lt --model RM1 --conduct st --ndatapoints 15000
mv src/outputs/lt/*.json src/outputs/lt/Experiment_17_RM1
mkdir src/outputs/lt/Experiment_18_RM2
python3 src/run_metric.py --data lt --model RM2 --conduct sh --ndatapoints 15000
python3 src/run_metric.py --data lt --model RM2 --conduct st --ndatapoints 15000
mv src/outputs/lt/*.json src/outputs/lt/Experiment_18_RM2
mkdir src/outputs/lt/Experiment_19_RSV
python3 src/run_metric.py --data lt --model RSV --conduct sh --ndatapoints 15000
python3 src/run_metric.py --data lt --model RSV --conduct st --ndatapoints 15000
mv src/outputs/lt/*.json src/outputs/lt/Experiment_19_RSV
mkdir src/outputs/lt/Experiment_20_RW
python3 src/run_metric.py --data lt --model RW --conduct sh --ndatapoints 15000
python3 src/run_metric.py --data lt --model RW --conduct st --ndatapoints 15000
mv src/outputs/lt/*.json src/outputs/lt/Experiment_20_RW
mkdir src/outputs/lt/Experiment_21_UIR
python3 src/run_metric.py --data lt --model UIR --conduct sh --ndatapoints 15000
python3 src/run_metric.py --data lt --model UIR --conduct st --ndatapoints 15000
mv src/outputs/lt/*.json src/outputs/lt/Experiment_21_UIR
