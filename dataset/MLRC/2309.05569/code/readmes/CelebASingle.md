# CelebA (single attributes)
This experiment compares the diversity and image quality of vanilla Stable Diffusion, ITI-GEN, HPS and HPS with negative prompting.
## Training
### Train ITI-GEN tokens
You need to run `jobfiles/celeba_single/iti_gen/train/${attribute}.sh`, for example `jobfiles/celeba_single/iti_gen/train/Young.sh`. To run all of them, do the following:
```shell
find jobfiles/celeba_single/iti_gen/train/*.sh | xargs -n1 bash
```
## Image generation
### Generate images with vanilla Stable Diffusion
```shell
python models/sd/scripts/txt2img.py \
    --config='models/sd/configs/stable-diffusion/v1-inference.yaml' \
    --ckpt='models/sd/models/ldm/stable-diffusion-v1/model.ckpt' \
    --plms \
    --prompt="a headshot of a person" \
    --negative_prompt="" \
    --outdir='results/celeba_single/vanilla' \
    --skip_grid \
    --n_iter=630 \>
    --n_samples=8 \
    --seed=42
```
### Generate images with ITI-GEN
You need to run `jobfiles/celeba_single/iti_gen/generation/${attribute}.sh`, for example `jobfiles/celeba_single/iti_gen/generation/Young.sh`. To run all of them, do the following
```shell
find jobfiles/celeba_single/iti_gen/generation/*.sh | xargs -n1 bash
```
### Generate images with HPS
You need to run `jobfiles/celeba_single/hps/generation/${attribute}.sh`, for example `jobfiles/celeba_single/hps/generation/Young.sh`. To run all of them, do the following
```shell
find jobfiles/celeba_single/hps/generation/*.sh | xargs -n1 bash
```
### Generate images with HPS with negative prompting
You need to run `jobfiles/celeba_single/hps_negative/generation/${attribute}.sh`, for example `jobfiles/celeba_single/hps_negative/generation/Young.sh`. To run all of them, do the following
```shell
find jobfiles/celeba_single/hps_negative/generation/*.sh | xargs -n1 bash
```
## Evaluation
### Compute FID score
Using the default parameters, 104 images are generated per category, making a total of $104 \times 80 = 8320$ images. In the original paper only 5K images are used, so we decide to take just 63 images per category, which adds up to $63 \times 80 = 5040$ images. 
```shell
python reduce_num_images.py
python fid.py --ffhq --folder1 "results/celeba_single/5K_iti_gen"
python fid.py --ffhq --folder1 "results/celeba_single/5K_hps"
python fid.py --ffhq --folder1 "results/celeba_single/5K_hps_negative"
python fid.py --ffhq --folder1 "results/celeba_single/vanilla"
```
### Compute KL-divergences
You need to run `jobfiles/celeba_single/${method}/evaluation/${attribute}.sh`, for example `jobfiles/celeba_single/hps_negative/evaluation/Young.sh`. To run all of them, do the following
```shell
find jobfiles/celeba_single -name "*.sh" | grep evaluation | xargs -n1 bash
```

