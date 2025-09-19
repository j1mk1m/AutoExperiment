# Studying How to Efficiently and Effectively Guide Models with Explanations - A Reproducibility Study

# TMLR/ MLRC Submission

## Setup
### Prerequisites
All the required packages can be installed using conda with the provided [environment.yml](environment.yml) file. 

### Data, Weights, and Model Training
Please refer to the [original repositiory by Rao et al. (2023)](https://github.com/sukrutrao/Model-Guidance?tab=readme-ov-file) for information on how to download the datasets and weights, as well as how to train the models.

### Waterbirds-100
Code for the experiments on the Waterbirds-100 dataset is not included in the original repository by Rao et al. (2023). Since this dataset is fairly small, all the necessary files are included in this repository for convenience under `datasets/Waterbirds-100`. In order to perform pre-processing, follow the instructions in the [README.md](datasets/) file in the `datasets/` directory.

There is a separate training script `train_waterbirds.py` for fine-tuning baseline and guided models on this dataset accross the two different tasks (1) conventional and (2) reversed. The command for executing this script follows the same pattern as the one for `train.py`, with a single additional flag for setting the task. Thus, to execute the script for the conventional task, you would use:

```
python train_waterbirds.py --dataset Waterbirds-100 --task birds ...
```

and for the reversed task:

```
python train_waterbirds.py --dataset Waterbirds-100 --task background ...
```

replacing `...` with other argument values, as when running `train.py`.

### Additional Files 
We extended the original repository by the following folders/files (in order of appearance in this repository):  
* [images/](images): folder containing the images used in the survey
* [result_jsons/](result_jsons): folder containing the final metric results from our experiments, stored in the json format
* [result_jsons_epg*/](result_jsons_epg*): folder containing the final metric results from our experiments on EPG*, stored in the json format
* [result_jsons_seg-vs-bb/](result_jsons_seg-vs-bb): folder containing the final metric results from our experiments on segmentation mask versus bounding box annotations, stored in the json format
* [result_jsons_seg_mask/](result_jsons_seg_mask): folder containing the final metric results from our experiments on segmentation mask annotation, stored in the json format
* [epg_vs_bbsize.csv](epg_vs_bbsize.csv): csv file, containing the data for evaluating the correlation between EPG score bounding box size
* [eval_baseline.py](eval_baseline.py), [eval_pareto.py](eval_pareto.py), [eval_segxepg.py](eval_segxepg.py): scripts for evaluating the saved checkpoints, and creating .json files with the results
* [get_qualitative_results.py](get_qualitative_results.py), [get_qualitative_results_dilation.py](get_qualitative_results_dilation.py): scripts for creating qualitative results from different checkpoints
* [qual_plotting_utils.py](qual_plotting_utils.py), [quant_plotting_utils.py](quant_plotting_utils.py): utils-functions for the [resulty.ipynb](results.ipynb) notebook, which can be used to easily reproduce the main results shown in our report
* [survey_plotting_results.py](survey_plotting_results.py): script for plotting the images used in the survey
* [survey_results.csv](survey_results.csv): csv file, containing the results from our survey in numerical form
* [test_bbox_size_epg.py](test_bbox_size_epg.py): script for obtaining the relevant data for the studying the correlation between bounding box size and EPG score
* [train_energy.py](train_energy.py), [train_seg.py](train_seg.py), [train_waterbird.py](train_waterbird.py): scripts for training models for the different experiments. The scripts follow the structure from the original train.py, with adjustments made to fit the corresponding experiment
* [visualization_function.py](visualization_function.py): script for creating qualitative results. Adapted from the original code by Rao et al. (2023), which they provided in response to our request
