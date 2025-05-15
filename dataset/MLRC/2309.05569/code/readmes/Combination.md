# Combination
This experiment combines ITI-GEN with HPSn.

## Train
```bash
python train_iti_gen.py \
    --prompt="a headshot of a person" \
    --attr-list="Age" \
    --epochs=30 \
    --save-ckpt-per-epochs=10

python train_iti_gen.py \
    --prompt="a headshot of a person" \
    --attr-list="Skin_tone" \
    --epochs=30 \
    --save-ckpt-per-epochs=10
```

## Prepend
This step is needed to modify the positive prompt
```bash
python prepend.py \
    --prompt='a headshot of a person' \
    --attr-list='Age' \
    --load-model-epoch=29 \
    --prepended-prompt='a headshot of a person with eyeglasses'

python prepend.py \
    --prompt='a headshot of a person' \
    --attr-list='Skin_tone' \
    --load-model-epoch=29 \
    --prepended-prompt='a headshot of a person with eyeglasses'
```

## Generation
```bash
python generation.py \
    --config="models/sd/configs/stable-diffusion/v1-inference.yaml" \
    --ckpt="models/sd/models/ldm/stable-diffusion-v1/model.ckpt" \
    --plms \
    --attr-list="Age" \
    --outdir="./results/combination/age/with-glasses" \
    --prompt-path="ckpts/a_headshot_of_a_person_Age/prepend_prompt_embedding_a_headshot_of_a_person_with_eyeglasses/basis_final_embed_29.pt" \
    --skip_grid \
    --n_iter=35 \
    --n_samples=8 \
    --seed 42

python generation.py \
    --config="models/sd/configs/stable-diffusion/v1-inference.yaml" \
    --ckpt="models/sd/models/ldm/stable-diffusion-v1/model.ckpt" \
    --plms \
    --attr-list="Age" \
    --outdir="./results/combination/age/no-glasses" \
    --prompt-path="ckpts/a_headshot_of_a_person_Age/original_prompt_embedding/basis_final_embed_29.pt" \
    --negative_prompt="eyeglasses" \
    --skip_grid \
    --n_iter=35 \
    --n_samples=8 \
    --seed 42

python generation.py \
    --config="models/sd/configs/stable-diffusion/v1-inference.yaml" \
    --ckpt="models/sd/models/ldm/stable-diffusion-v1/model.ckpt" \
    --plms \
    --attr-list="Skin_tone" \
    --outdir="./results/combination/skin-tone/with-glasses" \
    --prompt-path="ckpts/a_headshot_of_a_person_Skin_tone/prepend_prompt_embedding_a_headshot_of_a_person_with_eyeglasses/basis_final_embed_29.pt" \
    --skip_grid \
    --n_iter=53 \
    --n_samples=8 \
    --seed 42

python generation.py \
    --config="models/sd/configs/stable-diffusion/v1-inference.yaml" \
    --ckpt="models/sd/models/ldm/stable-diffusion-v1/model.ckpt" \
    --plms \
    --attr-list="Skin_tone" \
    --outdir="./results/combination/skin-tone/no-glasses" \
    --prompt-path="ckpts/a_headshot_of_a_person_Skin_tone/original_prompt_embedding/basis_final_embed_29.pt" \
    --negative_prompt="eyeglasses" \
    --skip_grid \
    --n_iter=53 \
    --n_samples=8 \
    --seed 42
```

## Evaluation
```bash
python evaluation.py \
    --img-folder "results/combination/age" \
    --class-list "a headshot of a person with eyeglasses" "a headshot of a person without eyeglasses"

python evaluation.py \
    --img-folder "results/combination/skin-tone" \
    --class-list "a headshot of a person with eyeglasses" "a headshot of a person without eyeglasses"
```