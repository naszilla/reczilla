<br/>
<p align="center"><img src="img/logo3.png" width=700 /></p>

----

The `RecZilla` codebase provides functionality to perform metalearning for algorithm selection on recommender systems datasets.

See our NeurIPS Datasets Track paper at https://arxiv.org/abs/2106.12543.

<p align="center"><img src="img/overview_figure.svg" width=700 /></p>


## Installation

TODO

To use our package, make sure you install all the dependencies in `requirements.txt` using 
```
pip install -r requirements.txt
```

## Sample Usage
A sample script to perform inference on a new dataset is provided in `run_reczilla_inference.sh`. It uses pre-trained Reczilla models (located in the folder `ReczillaModels`) to select and train a recommender on a dataset specified on a path. This script can be modified to run inference on new datasets.

The script `train_reczilla_models.sh` shows samples for training metalearners for different metrics.

---
## More details


The main script is `run_reczilla.py`, which must be run from RecSys2019_DeepLearning_Evaluation. It takes in these arguments:

```
> python -m ReczillaClassifier.run_reczilla -h
usage: run_reczilla.py [-h] [--train_meta] --metamodel_filepath
                       METAMODEL_FILEPATH
                       [--dataset_split_path DATASET_SPLIT_PATH]
                       [--rec_model_save_path REC_MODEL_SAVE_PATH]
                       [--metadataset_name METADATASET_NAME]
                       [--metamodel_name {xgboost,knn,linear,svm-poly}]
                       [--target_metric TARGET_METRIC]
                       [--num_algorithms NUM_ALGORITHMS]
                       [--num_metafeatures NUM_METAFEATURES]

Run Reczilla on a new dataset.

optional arguments:
  -h, --help            show this help message and exit
  --train_meta          Use to train a new metalearner Reczilla model (instead
                        of loading).
  --metamodel_filepath METAMODEL_FILEPATH
                        Filepath of Reczilla model (to save or load).
  --dataset_split_path DATASET_SPLIT_PATH
                        Path of dataset split to perform inference on. Only
                        required if performing inference
  --rec_model_save_path REC_MODEL_SAVE_PATH
                        Destination path for recommender model trained on
                        dataset on dataset_split_path.
  --metadataset_name METADATASET_NAME
                        Name of metadataset (required if training metamodel).
  --metamodel_name {xgboost,knn,linear,svm-poly}
                        Name of metalearner to use (required if training
                        metamodel).
  --target_metric TARGET_METRIC
                        Target metric to optimize.
  --num_algorithms NUM_ALGORITHMS
                        Number of algorithms to use in Reczilla (required if
                        training metamodel).
  --num_metafeatures NUM_METAFEATURES
                        Number of metafeatures to select for metalearner.
```

## Citation 
TODO

Please cite our work if you use code from this repo:
```bibtex
@inproceedings{xai-bench-2021,
  title={Synthetic Benchmarks for Scientific Research in Explainable Machine Learning}, 
  author={Liu, Yang and Khandagale, Sujay and White, Colin and Neiswanger, Willie}, 
  booktitle={Advances in Neural Information Processing Systems Datasets Track},
  year={2021}, 
} 
```
