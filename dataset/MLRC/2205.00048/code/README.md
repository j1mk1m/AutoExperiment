# JME-Fairness

This repository reproduces the paper [Joint Multisided Exposure Fairness for Recommendation](https://arxiv.org/abs/2205.00048) (SIGIR 2022) as part of the [ML Reproducibility Challenge 2022](https://paperswithcode.com/rc2022), and an extension of the JME-fairness framework to the neural Bert4Rec model.

![Fig5](Figures/Figure_5.png)
*The JME-Fairness metrics were calculated across a neural Bert4Rec model, for the MovieLens1M (orange) and the LibraryThing (turquoise) dataset.*

## Installation
Warning: Running setup will call for an installation of gdown in order to automatically retrieve the dataset. 

``` Installing and configuring repo
git clone https://github.com/OlineRanum/FACT.git
cd FACT
bash env/setup.sh
conda activate jme
```

## How to use the code
This section provides examples of how to use the repository

### Per experiment
How to run a single experiment
##### To run a single experiment with the default setting
```
python3 src/run_metric.py 
```
##### Produce the toy example experiment results
```
python3 src/utils/evaluation_functions/fariness_metrics.py
```

### To run all experiments associated with the reproduction study 
In the subsequent section, we provide instructions on running bash-scripts that reproduce all the fairness metric calculations associated with our paper, and how to plot 

#### Reproduce all plots in our reproduction study
To only reproduce the plots from our precalculated results (i.e. after running run_metric_<dataset>.sh), please use the following command. If you wish to run the full metric calculations, please proceed to the subsequent sections.

```
bash experiments/run_files/plot_reproduction_results.sh
```
#### Rerun all metric calculations on MovieLens1M dataset
NB! Expected runtime on RTX 3070 GPU is 13h - this will estimate the fairness metrics across all 21 pre-trained models to build the Kendall rank correlations for the MovieLens1M dataset. 

```
bash experiments/run_files/run_metrics_movielens.sh
```

#### Rerun all metric calculations on LibraryThing dataset
NB! Expected runtime on RTX 3070 GPU is 15h - this will estimate the fairness metrics across all 21 pre-trained models to build the Kendall rank correlations for the LibraryThing dataset. 

```
bash experiments/run_files/run_metrics_librarything.sh
```

### To plot the figures from your own reproduced results
```
bash experiments/run_files/plot_reestimated_results.sh
```

## Train BERT4Rec
For training and evaluating BERT4Rec, refer to the [BERT4Rec](https://github.com/OlineRanum/FACT/tree/main/src/BERT4Rec) directory 


## Directory Overview
### Folder
The [`env`](./env) contains the environment to run this codebase and a bash script _setup.sh_ to download the LibraryThing dataset and install/activate the environment. 

The [`experiments`](./experiments) contains the run files with all the bash scripts to reproduce all experiments associated with the reproduction study. 

The [`Figures`](./Figures) contains the cover picture.

The [`src`](./src) contains the source code of the project. 

## Datasets 

### MovieLens
MovieLens1M (ml‐1m) consists of 1,000,209 numerical ratings ranging from 1‐5. The reviews are supplied by 6,040 MovieLens users on 3,706 movies. The available metadata for the users includes 2 gender groups, 7 age groups, 21 occupation groups, and zip codes. The metadata for the movies comprises 19 genre groups (including unknown). Each user in the dataset has written at least 20 reviews.

The dataset is extracted from [here](https://grouplens.org/datasets/movielens/), please note that the dataset is already available in the repository.


### LibraryThing
LibraryThing (LT) Due to computational limitations we extract a subset of the LibraryThing dataset comprised of 702,522 reviews provided by 12,976 LibraryThing users on 325,075 books. The selection is made so that each user has written at least 20 reviews and each item is reviewed at least 3 times.

The dataset is extracted from [here](https://cseweb.ucsd.edu/~jmcauley/datasets.html), please note that the dataset is automatically downloaded when running the setup script.


## Results 

### Reproducibility study
*Claim 1*: Stochasticity impacts the JME‐fairness metrics and their corresponding disparity and relevance components, in the sense that there exists a disparity‐relevance trade‐off for each JME‐fairness metric.

We successfully reproduce the major trends of the JME-fairness metrics, although numerical deviations occur.

![Title](Figures/Figure_2.png)
![Title](Figures/Figure_3.png)
![Title](Figures/Table_2.png)

*Claim 2*: Each JME‐fairness metric quantifies a different notion of unfairness: a system that
performs well on one fairness dimension can be suboptimal for another.

We successfully reproduce the major trends of the JME-fairness metrics, although numerical deviations occur.

![Title](Figures/Figure_4.png)

*Claim 3*: Recommender systems can be optimized towards a specific fairness goal, based
on different combinations of JME‐fairness metrics.

The codebase does not contain code for demonstrating the third claim of the authors, as we were unable of training a converging model that included the JME-Fairness loss. As advised by the authors of the original paper, [this repository](https://github.com/jchanxtarov/bprmf) is suitable for further attempts to reconstruct the result of the third claim 



### Extended Research

We successfully demonstrate the application of the JME-Fairness curve to the neural model Bert4Rec trained on the MovieLens1M dataset. 

![Title](Figures/Figure_6.jpeg)

## Acknowledgements
* The authors of the original paper Wu et al., in particular, Dr. Bhaskar Mitra and Haloun Wu for supplying additional information regarding the pipeline and codebase of the original paper.
* [Template source](https://www.overleaf.com/project/62fcf81c144fd47671a39284)




## Citation of the original paper
```
@inproceedings{wu2022joint,
  title={Joint Multisided Exposure Fairness for Recommendation},
  author={Wu, Haolun and Mitra, Bhaskar and Ma, Chen and Diaz, Fernando and Liu, Xue},
  booktitle={SIGIR},
  publisher = {{ACM}},
  year={2022}
}
```

