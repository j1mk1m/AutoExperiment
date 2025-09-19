

# Diversity study on Eyeglasses dataset
This experiment compares the performance of ITI-GEN on the attribute combination Male x Eyeglasses when using different reference datasets for the Eyeglasses attribute.

## Create variations of the Eyeglasses dataset
```shell
python generate_jobfiles/diversity_eyeglasses/generate_datasets.py
```

## Training
### Train ITI-GEN tokens
```shell
python train_iti_gen.py \
    --prompt="a headshot of a person" \
    --attr-list="Male,Eyeglasses_non_diverse" \
    --epochs=30 \
    --save-ckpt-per-epochs=10
```
```shell
python train_iti_gen.py \
    --prompt="a headshot of a person" \
    --attr-list="Male,Eyeglasses_Male_only_positive" \
    --epochs=30 \
    --save-ckpt-per-epochs=10
```

## Generation
### Generate images with ITI-GEN
```shell
python generation.py \
    --config="models/sd/configs/stable-diffusion/v1-inference.yaml" \
    --ckpt="models/sd/models/ldm/stable-diffusion-v1/model.ckpt" \
    --plms \
    --attr-list="Male,Eyeglasses_non_diverse" \
    --outdir="results/diversity_eyeglasses/Male_Eyeglasses_non_diverse" \
    --prompt-path="ckpts/a_headshot_of_a_person_Male_Eyeglasses_non_diverse/original_prompt_embedding/basis_final_embed_29.pt" \
    --skip_grid \
    --n_iter=13 \
    --n_samples=8 \
    --seed=42
```
```shell
python generation.py \
    --config="models/sd/configs/stable-diffusion/v1-inference.yaml" \
    --ckpt="models/sd/models/ldm/stable-diffusion-v1/model.ckpt" \
    --plms \
    --attr-list="Male,Eyeglasses_Male_only_positive" \
    --outdir="results/diversity_eyeglasses/Male_Eyeglasses_Male_only_positive" \
    --prompt-path="ckpts/a_headshot_of_a_person_Male_Eyeglasses_Male_only_positive/original_prompt_embedding/basis_final_embed_29.pt" \
    --skip_grid \
    --n_iter=13 \
    --n_samples=8 \
    --seed=42
```

## Evaluation
### Compute KL Divergences
```shell
python evaluation.py \
    --img-folder "results/diversity_eyeglasses/Male_Eyeglasses_non_diverse" \
    --class-list "a headshot of an man with eyeglasses" "a headshot of a woman with eyeglasses" "a headshot of an man" "a headshot of a woman" 
```
```shell
python evaluation.py \
    --img-folder "results/diversity_eyeglasses/Male_Eyeglasses_Male_only_positive" \
    --class-list "a headshot of an man with eyeglasses" "a headshot of a woman with eyeglasses" "a headshot of an man" "a headshot of a woman" 

```