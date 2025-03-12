
# Multiple domains
This experiment demonstrates how ITI-GEN works on multiple domains.
## Training
### People
```shell
python train_iti_gen.py \
    --prompt="a headshot of a person" \
    --attr-list="Skin_tone" \
    --epochs=30 \
    --save-ckpt-per-epochs=10
```
### Natural scenes
```shell
python train_iti_gen.py \
    --prompt='a natural scene' \
    --attr-list='Colorful' \
    --epochs=30 \
    --save-ckpt-per-epochs=10
```
## Image generation
### People
```shell
python generation.py \
    --config="models/sd/configs/stable-diffusion/v1-inference.yaml" \
    --ckpt="models/sd/models/ldm/stable-diffusion-v1/model.ckpt" \
    --plms \
    --attr-list="Skin_tone" \
    --outdir="./results/multiple_domains/a_headshot_of_a_person_Skin_tone" \
    --prompt-path="ckpts/a_headshot_of_a_person_Skin_tone/original_prompt_embedding/basis_final_embed_29.pt" \
    --skip_grid \
    --n_iter=105  \
    --n_samples=8 \
    --seed 42
```
### Natural scenes
```shell
python generation.py \
    --config="models/sd/configs/stable-diffusion/v1-inference.yaml" \
    --ckpt="models/sd/models/ldm/stable-diffusion-v1/model.ckpt" \
    --plms \
    --attr-list="Colorful" \
    --outdir="./results/multiple_domains/a_natural_scene_Colorful" \
    --prompt-path="ckpts/a_natural_scene_Colorful/prepend_prompt_embedding_a_headshot_of_a_person/basis_final_embed_29.pt" \
    --skip_grid \
    --n_iter=126 \
    --n_samples=8 \
    --seed 42
```

## FID score
```shell
python fid.py \
    --folder1 "./results/multiple_domains/a_headshot_of_a_person_Skin_tone" \
    --ffhq 
python fid.py \
    --folder1 "./results/multiple_domains/a_natural_scene_Colorful" \
    --folder2 "./data/lhq" 
```