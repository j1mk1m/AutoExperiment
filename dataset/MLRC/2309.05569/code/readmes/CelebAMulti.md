
# CelebA (multiple attributes)
This experiment compares the diversity and image quality of vanilla Stable Diffusion, ITI-GEN, HPS and HPS with negative prompting.
## Training
### Train ITI-GEN tokens
You need to run `jobfiles/celeba_multi/${number_of_attributes}/iti_gen/train/${attribute_list}.sh`, for example `jobfiles/celeba_multi/3/iti_gen/train/Male_Young_Eyeglasses.sh`. To run all of them, do the following:
```shell
find jobfiles/celeba_multi/*/iti_gen/train/*.sh | xargs -n1 bash
```

### Generate images with ITI-GEN
You need to run `jobfiles/celeba_multi/${number_of_attributes}/iti_gen/generation/${attribute_list}.sh`, for example `jobfiles/celeba_multi/3/iti_gen/generation/Male_Young_Eyeglasses.sh`. To run all of them, do the following
```shell
find jobfiles/celeba_multi/*/iti_gen/generation/*.sh | xargs -n1 bash
```

### Generate images with HPS
You need to run `jobfiles/celeba_multi/${number_of_attributes}/hps/generation/${attribute_list}.sh`, for example `jobfiles/celeba_multi/3/hps/generation/Male_Young_Eyeglasses.sh`. To run all of them, do the following
```shell
find jobfiles/celeba_multi/*/hps/generation/*.sh | xargs -n1 bash
```

### Generate images with HPS with negative prompting
You need to run `jobfiles/celeba_multi/${number_of_attributes}/hps_negative/generation/${attribute_list}.sh`, for example `jobfiles/celeba_multi/3/hps_negative/generation/Male_Young_Eyeglasses.sh`. To run all of them, do the following
```shell
find jobfiles/celeba_multi/*/hps_negative/generation/*.sh | xargs -n1 bash
```

## Evaluation
### Compute KL-divergences
Using the default parameters, 104 images are generated per category.
You need to run `jobfiles/celeba_multi/${number_of_attributes}/${method}/evaluation/${attribute_list}.sh`, for example `jobfiles/celeba_multi/3/hps_negative/evaluation/Male_Young_Eyeglasses.sh`. To run all of them, do the following
```shell
find jobfiles/celeba_multi -name "*.sh" | grep evaluation | xargs -n1 bash
```
