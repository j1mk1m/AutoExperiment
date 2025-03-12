## Code structure
This folder contains all files to run the three experiments; the Reproducibility Experiment, the Model Agnosticism Experiment, and the Runtime Efficiency Experiment. Moreover, it includes notebooks to plot the experiment's results, like those displayed in the paper.  

The demo notebook provides a demo to explain the main components of using CartoonX for a CNN and CartoonX for a Vision Transformer. Moreover, it includes the script for creating an attention rollout for a Vision Transformer.

The following sections explain how to run the three different experiments (as mentioned in the paper) and to reproduce the results.

## The Reproducibility Experiment

To run this experiment for all 100 images, the following command is run in the base directory:
```
python cartoonx/main.py --imgdir=images/imagenet_sample --logdir=logs/experiment1 --n_images=100 
```
The relevant results are stored in the log directory. For every image, these results consist of a tensorboard log, a text file with the prediction, the npy object of the original image, as well as the CartoonX and the PixelRDE explanations. Moreover, the DWT/pixel mask of CartoonX/PixelRDE and a figure with the results are saved. To plot the figures used in the paper (i.e. Figure 1 and 2), `plot_exp1_qual_results.ipynb` is used.

Given a set of such results for different images, it is possible to generate the quantitative evaluation (i.e. Figure 3). For that, it is first necessary to calculate the effect on the distortion measure when continuously randomizing increasing fractions of the image components. This randomization procedure for the first two subplots is done for ...
1. Random pixels
2. Random wavelet components
3. PixelRDE
4. CartoonX

... by running the following command:

```
python experiments/distortion_curves.py --logdir=logs/experiment1 --storedir=logs/result1
```
The same logdir previously specified to store the explanation results has to be used. The results of this analysis are saved in the given storedir. Using `plot_quantitative_exp.ipynb`, the ultimate visualization can then be created. 

The third subplot requires explanations for different lambda values to be created. Given the resulting explanations for all desired lambda settings, it is possible to extract the distortion using:
```
python experiments/lambda_exploration.py --logdir=logs/experiment1_lambda --storedir=logs/result1
```
The creation of the visualizations is also part of the `plot_quantitative_exp.ipynb` notebook.

## The Model Agnosticism Experiment

To run this experiment for all 100 images, the following command is run in the main directory:
```
python experiments/model_agnosticism_exp.py --imgdir=images/imagenet_sample --logdir=logs/experiment2 --n_images=100 --lambda_vit=10
```

The qualitative results are stored in the log directory. For every image, it stores a tensorboard log, a text file containing the ViT and the CNN prediction, the npy object of the attention mask, the attention rollout, the CartoonX for ViT, the CartoonX for CNN, and the original image. Moreover, it stores the DWT masks or the CNN and the ViT. Lastly, it saves a figure with all the results (i.e. the CartoonX for ViT and CNN and the attention rollout). The figures of the qualitative results from the paper (i.e. figure 6) are thereafter obtained by running the `plot_exp2_qual_results.ipynb` notebook.

Similarly to the quantitative evaluation of the Reproducibility Experiment, it is also necessary to first generate the additional measurements using the following command: 
```
python experiments/distortion_curves.py --logdir=logs/experiment2 --storedir=logs/result1 --experiment_type ViT
```
These thereby obtained results will be, likewise, plotted in the latter cells of the `plot_quantitative_exp.ipynb` notebook.

## The Runtime Efficiency Experiment

To run this experiment, the following command is run in the main directory four times. Everytime, the used initialization method is varied. The initialization method is specified in the `hparams.yaml` file and is either `ones`, `foreground` or `random`. Additionally, the flag `--preoptimize` can be set to enable the smart initialization heuristic either `ones` or `foreground`.
```
python cartoonx/main.py --imgdir=images/imagenet_sample --logdir=logs/experiment3 --n_images=2
```
The effect of the different initialization methods on the runtime performance of CartoonX was assessed based on the loss curves that are available using the tensorboard logs. 

