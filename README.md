<br/>
<p align="center"><img src="img/logo3.png" width=700 /></p>

----
![Crates.io](https://img.shields.io/crates/l/Ap)

`RecZilla` is a framework which provides the functionality to perform metalearning for algorithm selection on recommender systems datasets. It uses a meta-learner model to predict the best algorithm and hyperparameters for new, unseen datasets. 

## Overview
The figure below shows the overview of the end-to-end `RecZilla` framework pipeline.

<p align="center"><img src="img/reczilla_overview.png" width=700 /></p>


## Installation

You need Python 3.6 to use this repository.

You can start by first creating a new environment using `conda` or your preferred method.

```
# using conda
conda create -n DLevaluation python=3.6 anaconda
conda activate DLevaluation
```

Once you're done with the above step, you need to install all the dependencies in the `requirements.txt` file using,
```
pip install -r requirements.txt
```

Next step, you need to compile all the Cython algorithms. For that you will need to install `gcc` and `python3-dev`. You can install it on Linux as,
```
sudo apt install gcc 
sudo apt-get install python3-dev
```

Once installed, you can compile all the Cython algorithms by running the below command in the `RecSys2019_DeepLearning_Evaluation` directory,
```
python run_compile_all_cython.py
```
And, you're all setup!

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

