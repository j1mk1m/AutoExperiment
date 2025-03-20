
# Train-once-for-all
This experiment shows the plug-and-play capabilities of ITI-GEN: the text prompt used for training need not be the same as the one used for generation.

## Train
### Young
Note that Young is a binary feature, not to be confused with Age.
```shell
python train_iti_gen.py \
    --prompt="a headshot of a person" \
    --attr-list="Young" \
    --epochs=30 \
    --save-ckpt-per-epochs=10
```
### Male
```shell
python train_iti_gen.py \
    --prompt="a headshot of a person" \
    --attr-list="Male" \
    --epochs=30 \
    --save-ckpt-per-epochs=10
```

## Preprend
This step will append the inclusive tokens to a different text prompt.

### Firefighter ++ Young
```shell
python prepend.py \
    --prompt='a headshot of a person' \
    --attr-list='Young' \
    --load-model-epoch=29 \
    --prepended-prompt='a headshot of a firefighter'
```
### Doctor ++ Male
```shell
python prepend.py \
    --prompt="a headshot of a person" \
    --attr-list="Male" \
    --load-model-epoch=29 \
    --prepended-prompt="a headshot of a doctor"
```

## Generate images

### Firefighter ++ Young
```shell
python generation.py \
    --config="models/sd/configs/stable-diffusion/v1-inference.yaml" \
    --ckpt="models/sd/models/ldm/stable-diffusion-v1/model.ckpt" \
    --plms \
    --attr-list="Young" \
    --outdir="./results/train_once_for_all/firefighter" \
    --prompt-path="ckpts/a_headshot_of_a_person_Young/prepend_prompt_embedding_a_headshot_of_a_firefighter/basis_final_embed_29.pt" \
    --skip_grid \
    --n_iter=3 \
    --n_samples=4 \
    --seed 42
```
### Doctor ++ Male
```shell
python generation.py \
    --config="models/sd/configs/stable-diffusion/v1-inference.yaml" \
    --ckpt="models/sd/models/ldm/stable-diffusion-v1/model.ckpt" \
    --plms \
    --attr-list="Male" \
    --outdir="./results/train_once_for_all/doctor" \
    --prompt-path="ckpts/a_headshot_of_a_person_Male/prepend_prompt_embedding_a_headshot_of_a_doctor/basis_final_embed_29.pt" \
    --skip_grid \
    --n_iter=3 \
    --n_samples=4 \
    --seed 42
```
