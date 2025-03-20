## Downloading and Pre-processing Datasets

For each dataset, use the provided preprocessing script for each of the train, validation, and test scripts as follows:

### VOC2007 and COCO2014

```bash
python preprocess.py --split train
python preprocess.py --split val
python preprocess.py --split test
```

For COCO2014, the dataset needs to be first downloaded using [download.sh](COCO2014/download.sh).

### Waterbirds-100

There are two tasks performed on this dataset and the preprocessing has to be done separately for each task:
1) Classifying birds (Landbird vs Waterbird)
   
   ```
   python preprocess.py --data_root ./data --task birds --split train
   python preprocess.py --data_root ./data --task birds --split val
   python preprocess.py --data_root ./data --task birds --split test
   ```
3) Classifying background (Land vs Water)
   
   ```
   python preprocess.py --data_root ./data --task background --split train
   python preprocess.py --data_root ./data --task background --split val
   python preprocess.py --data_root ./data --task background --split test
   ```

### Acknowledgements

The scripts provided here build upon scripts from [stevenstalder/NN-Explainer](https://github.com/stevenstalder/NN-Explainer).


