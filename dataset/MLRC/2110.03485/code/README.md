# On the Reproducibility of CartoonX


<p align="center">
<strong>Authors:</strong> Elias Dubbeldam, Aniek Eijpe, Jona Ruthardt, Robin Sasse <br>
<a href="https://doi.org/10.5281/zenodo.8173672" style="background-color: #4CAF50; color: white; padding: 10px 20px; text-align: center; text-decoration: none; display: inline-block; border-radius: 5px;">Link to Paper</a>
</p>



## Description
This repository contains the code of *On the Reproducibility of CartoonX*. 
This project is a reproducibility study of the results presented in *Cartoon Explanations of Image Classifiers*[^1], where they introduced CartoonX (Cartoon Explanations).
CartoonX is a rate‐distortion‐based explanation method for image classifiers operating in the wavelet domain to identify the components (i.e. wavelet coefficients) of an image that are most decisive for the model’s prediction. 

The contents of this repository include the required implementation for reproducing the original results and additional extensions. More information on the exact reproducibility and extension experiments can be found in the `experiments` folder. 

## Code structure
- `cartoonx` source code [^2] that generates CartoonX and Pixel RDE explanations. This code was adapted for our experiments.
- `experiments` code to run, explain and visualize the experiments. 
- `images` folders containing the images that are being explained.
- `logs` location where data of the different experiments is stored.
- `results` results underlying qualitative analysis presented in paper

## Installation
```
# clone project
git clone https://github.com/JonaRuthardt/MLRC-CartoonX.git

# Enter directory
cd MLRC-CartoonX

# Create and conda virtual environment
conda create --name cartoonx python=3.8.8

# if on Windows/Mac
conda activate cartoonx 

# if on Linux
source activate cartoonx 

# install other project dependencies from requirements file   
pip install -r requirements.txt

# install pytorch wavelets package 
git clone https://github.com/fbcotter/pytorch_wavelets
cd pytorch_wavelets
pip install .
pip install -r tests/requirements.txt
pytest tests/ # Unit tests

# install vit-explain package in the main directory
cd ..
git clone https://github.com/jacobgil/vit-explain.git
``` 

## How to run
Information on how to run the experiments can be found in a separate `README.md` in the `experiments` folder.

[^1]: Kolek et al. (2022, October). *Cartoon Explanations of Image Classifiers.* In Computer Vision–ECCV 2022.
[^2]: https://github.com/skmda37/CartoonX.git
