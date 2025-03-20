echo Experiment 1
python train.py --dataset VOC2007 --model_backbone bcos --total_epochs 1 --localization_loss_fn Energy --attribution_method BCos --optimize_explanations
echo Experiment 2
python train.py --dataset VOC2007 --model_backbone bcos --total_epochs 1 --localization_loss_fn Energy --attribution_method GradCam --optimize_explanations
echo Experiment 3
python train.py --dataset VOC2007 --model_backbone bcos --total_epochs 1 --localization_loss_fn Energy --attribution_method IxG --optimize_explanations
