echo Experiment 1
bash jobfiles/celeba_single/iti_gen/train/5_o_Clock_Shadow.sh
python generation.py \
    --config="models/sd/configs/stable-diffusion/v1-inference.yaml" \
    --ckpt="models/sd/models/ldm/stable-diffusion-v1/model.ckpt" \
    --plms \
    --attr-list="5_o_Clock_Shadow" \
    --outdir="results/celeba_single/iti_gen/5_o_Clock_Shadow" \
    --prompt-path="ckpts/a_headshot_of_a_person_5_o_Clock_Shadow/original_prompt_embedding/basis_final_embed_4.pt" \
    --skip_grid \
    --n_iter=1 \
    --n_samples=4 \
    --seed=0
bash jobfiles/celeba_single/iti_gen/evaluation/5_o_Clock_Shadow.sh
echo Experiment 2
bash jobfiles/celeba_single/iti_gen/train/High_Cheekbones.sh
python generation.py \
    --config="models/sd/configs/stable-diffusion/v1-inference.yaml" \
    --ckpt="models/sd/models/ldm/stable-diffusion-v1/model.ckpt" \
    --plms \
    --attr-list="High_Cheekbones" \
    --outdir="results/celeba_single/iti_gen/High_Cheekbones" \
    --prompt-path="ckpts/a_headshot_of_a_person_High_Cheekbones/original_prompt_embedding/basis_final_embed_4.pt" \
    --skip_grid \
    --n_iter=1 \
    --n_samples=4 \
    --seed=19
bash jobfiles/celeba_single/iti_gen/evaluation/High_Cheekbones.sh
echo Experiment 3
bash jobfiles/celeba_single/iti_gen/train/Bangs.sh
echo Experiment 4
bash jobfiles/celeba_single/iti_gen/train/Chubby.sh
echo Experiment 5
bash jobfiles/celeba_single/iti_gen/train/Smiling.sh
echo Experiment 6
bash jobfiles/celeba_single/iti_gen/train/Sideburns.sh
echo Experiment 7
bash jobfiles/celeba_multi/2/iti_gen/train/Male_Young.sh
echo Experiment 8
bash jobfiles/celeba_multi/2/iti_gen/train/Male_Young.sh
python generation.py \
    --config="models/sd/configs/stable-diffusion/v1-inference.yaml" \
    --ckpt="models/sd/models/ldm/stable-diffusion-v1/model.ckpt" \
    --plms \
    --attr-list="Male,Young" \
    --outdir="results/celeba_multi/2/iti_gen/Male_Young" \
    --prompt-path="ckpts/a_headshot_of_a_person_Male_Young/original_prompt_embedding/basis_final_embed_4.pt" \
    --skip_grid \
    --n_iter=1 \
    --n_samples=4 \
    --seed=0
bash jobfiles/celeba_multi/2/iti_gen/evaluation/Male_Young.sh
echo Experiment 9
bash jobfiles/celeba_multi/3/iti_gen/train/Male_Young_Eyeglasses.sh
echo Experiment 10
bash jobfiles/celeba_multi/4/iti_gen/train/Male_Young_Eyeglasses_Smiling.sh
