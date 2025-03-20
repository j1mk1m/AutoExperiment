# Initial configuration
## Download weights
Download [Stable Diffusion weights](https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4.ckpt) and save them as `models/sd/models/ldm/stable-diffusion-v1/model.ckpt` (create the `stable-diffusion-v1` folder if needed).
## Download datasets
|   Dataset    |      Description      |       Attribute Used        |                                        Google Drive                                        |
|:------------:|:---------------------:|:---------------------------:|:------------------------------------------------------------------------------------------:|
|  [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)  |   Real face images    | 40 binary facial attributes | [Link](https://drive.google.com/file/d/1_wxcrzirofEge4i8LTyYBAL0SMQ_LwGO/view?usp=sharing) | 
| [FairFace](https://github.com/joojs/fairface) |   Real face images    |    Age with 9 categories    | [Link](https://drive.google.com/file/d/1_xtui0b0O52u38jbJzrxW8yRRiBHnZaA/view?usp=sharing) |
|   [FAIR](https://trust.is.tue.mpg.de/)   | Synthetic face images |   Skin tone with 6 categories    | [Link](https://drive.google.com/file/d/1_wiqq7FDByLp8Z4WQOeboSEXYsCzmV76/view?usp=sharing) |
|   [LHQ](https://universome.github.io/alis)    |    Natural scenes     | 11 global scene attributes  | [Link](https://drive.google.com/file/d/1_ypk4ouxQptBevUTcWSp0ZbxvqSZGiKg/view?usp=sharing) |

Download the above datasets and unzip them as follows
```angular2html
|-- data
|   |-- celeba
|   |   |-- 5_o_Clock_Shadow
|   |   |-- Bald
|   |   |-- ...

|   |-- FAIR_benchmark
|   |   |-- Skin_tone

|   |-- fairface
|   |   |-- Age

|   |-- lhq
|   |   |-- Bright
|   |   |-- Colorful
|   |   |-- ...
```

## Configure environment
```shell
conda env create --name iti-gen --file=environment.yml
source activate iti-gen
cd models/sd/
pip install -e .
cd ../../
```
# Experiments
All of our experiments are reproducible, and we include bash scripts to run them in the `jobfiles` folder. The bash scripts are generated automatically by Python scripts that live in the `generate_jobfiles` folder. You might be interested in modifying the script-creating scripts (e.g. to include SLURM directives, modify the batch size, etc.). If you do so, you can re-generate the bash scripts by running 
```shell
find generate_jobfiles/ -name "*.py" | xargs -n1 python
```
## CelebA (single attributes)
This experiment compares the diversity and image quality of vanilla Stable Diffusion, ITI-GEN, HPS and HPS with negative prompting. It computes the FID score and KL divergences (using CLIP as classifier) for all 40 attributes of CelebA. See instructions [here](readmes/CelebASingle.md).
## Multiple domains
This experiment demonstrates how ITI-GEN works on multiple domains, such as headshots and landscapes. See instructions [here](readmes/MultipleDomains.md).
## Train-once-for-all
This experiment shows the plug-and-play capabilities of ITI-GEN: the text prompt used for training need not be the same as the one used for generation. We train "Age" and "Gender" using "a headshot of a person", and apply them using "a headshot of a firefighter" and "a headshot of a doctor", respectively. See instructions [here](readmes/TrainOnceForAll.md).
## ControlNet
This experiment proves the plug-and-play capabilities of ITI-GEN when it is used with ControlNet. See instructions [here](readmes/ControlNet.md).
## Reference size
This experiment varies the number of reference images per category to prove that ITI-GEN is robust in the low-data regime. See instructions [here](readmes/ReferenceSize.md).
## Proxy features
This experiment shows how ITI-GEN uses gender as a proxy attribute when learning other attributes, such as "Bald", "Mustache" or "Beard". See instructions [here](readmes/ProxyFeatures.md).
## CelebA (multiple attributes)
This experiment compares the diversity and image quality of vanilla Stable Diffusion, ITI-GEN, HPS and HPS with negative prompting. It generates images for some attribute combinations and computes the KL divergence (using CLIP as classifier).  See instructions [here](readmes/CelebAMulti.md).
## Eyeglasses
This experiment compares the performance of ITI-GEN on the attribute combination Male x Eyeglasses when using different reference datasets for the Eyeglasses attribute. See instructions [here](readmes/DiversityEyeglasses.md)
## Benchmark
This experiments shows that the training time is proportional to the size of the joint distribution, that grows exponentially with the number of attributes. See instructions[here](readmes/Benchmark.md)
## Combination
This experiment combines ITI-GEN with HPSn. See instructions [here](readmes/Combination.md)