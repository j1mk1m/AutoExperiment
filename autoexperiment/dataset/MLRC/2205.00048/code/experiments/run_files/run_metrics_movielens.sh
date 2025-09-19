mkdir src/outputs/ml-1m/Experiment_1_BPRMF
mkdir src/outputs/ml-1m/Experiment_1_BPRMF/Gender
mkdir src/outputs/ml-1m/Experiment_1_BPRMF/Age
python3 src/run_metric.py --data ml-1m --model BPRMF --conduct sh --age N
python3 src/run_metric.py --data ml-1m --model BPRMF --conduct st --age N
mv src/outputs/ml-1m/*.json src/outputs/ml-1m/Experiment_1_BPRMF/Gender
python3 src/run_metric.py --data ml-1m --model BPRMF --conduct sh --age Y
python3 src/run_metric.py --data ml-1m --model BPRMF --conduct st --age Y
mv src/outputs/ml-1m/*.json src/outputs/ml-1m/Experiment_1_BPRMF/Age

mkdir src/outputs/ml-1m/Experiment_2_LDA
mkdir src/outputs/ml-1m/Experiment_2_LDA/Gender
mkdir src/outputs/ml-1m/Experiment_2_LDA/Age
python3 src/run_metric.py --data ml-1m --model LDA --conduct sh --age N
python3 src/run_metric.py --data ml-1m --model LDA --conduct st --age N
mv src/outputs/ml-1m/*.json src/outputs/ml-1m/Experiment_2_LDA/Gender
python3 src/run_metric.py --data ml-1m --model LDA --conduct sh --age Y
python3 src/run_metric.py --data ml-1m --model LDA --conduct st --age Y
mv src/outputs/ml-1m/*.json src/outputs/ml-1m/Experiment_2_LDA/Age

mkdir src/outputs/ml-1m/Experiment_3_PureSVD
mkdir src/outputs/ml-1m/Experiment_3_PureSVD/Gender
mkdir src/outputs/ml-1m/Experiment_3_PureSVD/Age
python3 src/run_metric.py --data ml-1m --model PureSVD --conduct sh --age N
python3 src/run_metric.py --data ml-1m --model PureSVD --conduct st --age N
mv src/outputs/ml-1m/*.json src/outputs/ml-1m/Experiment_3_PureSVD/Gender
python3 src/run_metric.py --data ml-1m --model PureSVD --conduct sh --age Y
python3 src/run_metric.py --data ml-1m --model PureSVD --conduct st --age Y
mv src/outputs/ml-1m/*.json src/outputs/ml-1m/Experiment_3_PureSVD/Age

mkdir src/outputs/ml-1m/Experiment_4_SLIM
mkdir src/outputs/ml-1m/Experiment_4_SLIM/Gender
mkdir src/outputs/ml-1m/Experiment_4_SLIM/Age
python3 src/run_metric.py --data ml-1m --model SLIM --conduct sh --age N
python3 src/run_metric.py --data ml-1m --model SLIM --conduct st --age N
mv src/outputs/ml-1m/*.json src/outputs/ml-1m/Experiment_4_SLIM/Gender
python3 src/run_metric.py --data ml-1m --model SLIM --conduct sh --age Y
python3 src/run_metric.py --data ml-1m --model SLIM --conduct st --age Y
mv src/outputs/ml-1m/*.json src/outputs/ml-1m/Experiment_4_SLIM/Age

mkdir src/outputs/ml-1m/Experiment_5_WRMF
mkdir src/outputs/ml-1m/Experiment_5_WRMF/Gender
mkdir src/outputs/ml-1m/Experiment_5_WRMF/Age
python3 src/run_metric.py --data ml-1m --model WRMF --conduct sh --age N
python3 src/run_metric.py --data ml-1m --model WRMF --conduct st --age N
mv src/outputs/ml-1m/*.json src/outputs/ml-1m/Experiment_5_WRMF/Gender
python3 src/run_metric.py --data ml-1m --model WRMF --conduct sh --age Y
python3 src/run_metric.py --data ml-1m --model WRMF --conduct st --age Y
mv src/outputs/ml-1m/*.json src/outputs/ml-1m/Experiment_5_WRMF/Age

mkdir src/outputs/ml-1m/Experiment_6_CHI2
mkdir src/outputs/ml-1m/Experiment_6_CHI2/Gender
mkdir src/outputs/ml-1m/Experiment_6_CHI2/Age
python3 src/run_metric.py --data ml-1m --model CHI2 --conduct sh --age N
python3 src/run_metric.py --data ml-1m --model CHI2 --conduct st --age N
mv src/outputs/ml-1m/*.json src/outputs/ml-1m/Experiment_6_CHI2/Gender
python3 src/run_metric.py --data ml-1m --model CHI2 --conduct sh --age Y
python3 src/run_metric.py --data ml-1m --model CHI2 --conduct st --age Y
mv src/outputs/ml-1m/*.json src/outputs/ml-1m/Experiment_6_CHI2/Age

mkdir src/outputs/ml-1m/Experiment_7_HT
mkdir src/outputs/ml-1m/Experiment_7_HT/Gender
mkdir src/outputs/ml-1m/Experiment_7_HT/Age
python3 src/run_metric.py --data ml-1m --model HT --conduct sh --age N
python3 src/run_metric.py --data ml-1m --model HT --conduct st --age N
mv src/outputs/ml-1m/*.json src/outputs/ml-1m/Experiment_7_HT/Gender
python3 src/run_metric.py --data ml-1m --model HT --conduct sh --age Y
python3 src/run_metric.py --data ml-1m --model HT --conduct st --age Y
mv src/outputs/ml-1m/*.json src/outputs/ml-1m/Experiment_7_HT/Age

mkdir src/outputs/ml-1m/Experiment_8_KLD
mkdir src/outputs/ml-1m/Experiment_8_KLD/Gender
mkdir src/outputs/ml-1m/Experiment_8_KLD/Age
python3 src/run_metric.py --data ml-1m --model KLD --conduct sh --age N
python3 src/run_metric.py --data ml-1m --model KLD --conduct st --age N
mv src/outputs/ml-1m/*.json src/outputs/ml-1m/Experiment_8_KLD/Gender
python3 src/run_metric.py --data ml-1m --model KLD --conduct sh --age Y
python3 src/run_metric.py --data ml-1m --model KLD --conduct st --age Y
mv src/outputs/ml-1m/*.json src/outputs/ml-1m/Experiment_8_KLD/Age

mkdir src/outputs/ml-1m/Experiment_9_LMWI
mkdir src/outputs/ml-1m/Experiment_9_LMWI/Gender
mkdir src/outputs/ml-1m/Experiment_9_LMWI/Age
python3 src/run_metric.py --data ml-1m --model LMWI --conduct sh --age N
python3 src/run_metric.py --data ml-1m --model LMWI --conduct st --age N
mv src/outputs/ml-1m/*.json src/outputs/ml-1m/Experiment_9_LMWI/Gender
python3 src/run_metric.py --data ml-1m --model LMWI --conduct sh --age Y
python3 src/run_metric.py --data ml-1m --model LMWI --conduct st --age Y
mv src/outputs/ml-1m/*.json src/outputs/ml-1m/Experiment_9_LMWI/Age

mkdir src/outputs/ml-1m/Experiment_10_LMWU
mkdir src/outputs/ml-1m/Experiment_10_LMWU/Gender
mkdir src/outputs/ml-1m/Experiment_10_LMWU/Age
python3 src/run_metric.py --data ml-1m --model LMWU --conduct sh --age N
python3 src/run_metric.py --data ml-1m --model LMWU --conduct st --age N
mv src/outputs/ml-1m/*.json src/outputs/ml-1m/Experiment_10_LMWU/Gender
python3 src/run_metric.py --data ml-1m --model LMWU --conduct sh --age Y
python3 src/run_metric.py --data ml-1m --model LMWU --conduct st --age Y
mv src/outputs/ml-1m/*.json src/outputs/ml-1m/Experiment_10_LMWU/Age

mkdir src/outputs/ml-1m/Experiment_11_SVD
mkdir src/outputs/ml-1m/Experiment_11_SVD/Gender
mkdir src/outputs/ml-1m/Experiment_11_SVD/Age
python3 src/run_metric.py --data ml-1m --model SVD --conduct sh --age N
python3 src/run_metric.py --data ml-1m --model SVD --conduct st --age N
mv src/outputs/ml-1m/*.json src/outputs/ml-1m/Experiment_11_SVD/Gender
python3 src/run_metric.py --data ml-1m --model SVD --conduct sh --age Y
python3 src/run_metric.py --data ml-1m --model SVD --conduct st --age Y
mv src/outputs/ml-1m/*.json src/outputs/ml-1m/Experiment_11_SVD/Age

mkdir src/outputs/ml-1m/Experiment_12_NNI
mkdir src/outputs/ml-1m/Experiment_12_NNI/Gender
mkdir src/outputs/ml-1m/Experiment_12_NNI/Age
python3 src/run_metric.py --data ml-1m --model NNI --conduct sh --age N
python3 src/run_metric.py --data ml-1m --model NNI --conduct st --age N
mv src/outputs/ml-1m/*.json src/outputs/ml-1m/Experiment_12_NNI/Gender
python3 src/run_metric.py --data ml-1m --model NNI --conduct sh --age Y
python3 src/run_metric.py --data ml-1m --model NNI --conduct st --age Y
mv src/outputs/ml-1m/*.json src/outputs/ml-1m/Experiment_12_NNI/Age

mkdir src/outputs/ml-1m/Experiment_13_NNU
mkdir src/outputs/ml-1m/Experiment_13_NNU/Gender
mkdir src/outputs/ml-1m/Experiment_13_NNU/Age
python3 src/run_metric.py --data ml-1m --model NNU --conduct sh --age N
python3 src/run_metric.py --data ml-1m --model NNU --conduct st --age N
mv src/outputs/ml-1m/*.json src/outputs/ml-1m/Experiment_13_NNU/Gender
python3 src/run_metric.py --data ml-1m --model NNU --conduct sh --age Y
python3 src/run_metric.py --data ml-1m --model NNU --conduct st --age Y
mv src/outputs/ml-1m/*.json src/outputs/ml-1m/Experiment_13_NNU/Age

mkdir src/outputs/ml-1m/Experiment_14_PLSA
mkdir src/outputs/ml-1m/Experiment_14_PLSA/Gender
mkdir src/outputs/ml-1m/Experiment_14_PLSA/Age
python3 src/run_metric.py --data ml-1m --model PLSA --conduct sh --age N
python3 src/run_metric.py --data ml-1m --model PLSA --conduct st --age N
mv src/outputs/ml-1m/*.json src/outputs/ml-1m/Experiment_14_PLSA/Gender
python3 src/run_metric.py --data ml-1m --model PLSA --conduct sh --age Y
python3 src/run_metric.py --data ml-1m --model PLSA --conduct st --age Y
mv src/outputs/ml-1m/*.json src/outputs/ml-1m/Experiment_14_PLSA/Age

mkdir src/outputs/ml-1m/Experiment_15_POP
mkdir src/outputs/ml-1m/Experiment_15_POP/Gender
mkdir src/outputs/ml-1m/Experiment_15_POP/Age
python3 src/run_metric.py --data ml-1m --model POP --conduct sh --age N
python3 src/run_metric.py --data ml-1m --model POP --conduct st --age N
mv src/outputs/ml-1m/*.json src/outputs/ml-1m/Experiment_15_POP/Gender
python3 src/run_metric.py --data ml-1m --model POP --conduct sh --age Y
python3 src/run_metric.py --data ml-1m --model POP --conduct st --age Y
mv src/outputs/ml-1m/*.json src/outputs/ml-1m/Experiment_15_POP/Age

mkdir src/outputs/ml-1m/Experiment_16_Random
mkdir src/outputs/ml-1m/Experiment_16_Random/Gender
mkdir src/outputs/ml-1m/Experiment_16_Random/Age
python3 src/run_metric.py --data ml-1m --model Random --conduct sh --age N
python3 src/run_metric.py --data ml-1m --model Random --conduct st --age N
mv src/outputs/ml-1m/*.json src/outputs/ml-1m/Experiment_16_Random/Gender
python3 src/run_metric.py --data ml-1m --model Random --conduct sh --age Y
python3 src/run_metric.py --data ml-1m --model Random --conduct st --age Y
mv src/outputs/ml-1m/*.json src/outputs/ml-1m/Experiment_16_Random/Age

mkdir src/outputs/ml-1m/Experiment_17_RM1
mkdir src/outputs/ml-1m/Experiment_17_RM1/Gender
mkdir src/outputs/ml-1m/Experiment_17_RM1/Age
python3 src/run_metric.py --data ml-1m --model RM1 --conduct sh --age N
python3 src/run_metric.py --data ml-1m --model RM1 --conduct st --age N
mv src/outputs/ml-1m/*.json src/outputs/ml-1m/Experiment_17_RM1/Gender
python3 src/run_metric.py --data ml-1m --model RM1 --conduct sh --age Y
python3 src/run_metric.py --data ml-1m --model RM1 --conduct st --age Y
mv src/outputs/ml-1m/*.json src/outputs/ml-1m/Experiment_17_RM1/Age

mkdir src/outputs/ml-1m/Experiment_18_RM2
mkdir src/outputs/ml-1m/Experiment_18_RM2/Gender
mkdir src/outputs/ml-1m/Experiment_18_RM2/Age
python3 src/run_metric.py --data ml-1m --model RM2 --conduct sh --age N
python3 src/run_metric.py --data ml-1m --model RM2 --conduct st --age N
mv src/outputs/ml-1m/*.json src/outputs/ml-1m/Experiment_18_RM2/Gender
python3 src/run_metric.py --data ml-1m --model RM2 --conduct sh --age Y
python3 src/run_metric.py --data ml-1m --model RM2 --conduct st --age Y
mv src/outputs/ml-1m/*.json src/outputs/ml-1m/Experiment_18_RM2/Age

mkdir src/outputs/ml-1m/Experiment_19_RSV
mkdir src/outputs/ml-1m/Experiment_19_RSV/Gender
mkdir src/outputs/ml-1m/Experiment_19_RSV/Age
python3 src/run_metric.py --data ml-1m --model RSV --conduct sh --age N
python3 src/run_metric.py --data ml-1m --model RSV --conduct st --age N
mv src/outputs/ml-1m/*.json src/outputs/ml-1m/Experiment_19_RSV/Gender
python3 src/run_metric.py --data ml-1m --model RSV --conduct sh --age Y
python3 src/run_metric.py --data ml-1m --model RSV --conduct st --age Y
mv src/outputs/ml-1m/*.json src/outputs/ml-1m/Experiment_19_RSV/Age

mkdir src/outputs/ml-1m/Experiment_20_RW
mkdir src/outputs/ml-1m/Experiment_20_RW/Gender
mkdir src/outputs/ml-1m/Experiment_20_RW/Age
python3 src/run_metric.py --data ml-1m --model RW --conduct sh --age N
python3 src/run_metric.py --data ml-1m --model RW --conduct st --age N
mv src/outputs/ml-1m/*.json src/outputs/ml-1m/Experiment_20_RW/Gender
python3 src/run_metric.py --data ml-1m --model RW --conduct sh --age Y
python3 src/run_metric.py --data ml-1m --model RW --conduct st --age Y
mv src/outputs/ml-1m/*.json src/outputs/ml-1m/Experiment_20_RW/Age

mkdir src/outputs/ml-1m/Experiment_21_UIR
mkdir src/outputs/ml-1m/Experiment_21_UIR/Gender
mkdir src/outputs/ml-1m/Experiment_21_UIR/Age
python3 src/run_metric.py --data ml-1m --model UIR --conduct sh --age N
python3 src/run_metric.py --data ml-1m --model UIR --conduct st --age N
mv src/outputs/ml-1m/*.json src/outputs/ml-1m/Experiment_21_UIR/Gender
python3 src/run_metric.py --data ml-1m --model UIR --conduct sh --age Y
python3 src/run_metric.py --data ml-1m --model UIR --conduct st --age Y
mv src/outputs/ml-1m/*.json src/outputs/ml-1m/Experiment_21_UIR/Age

