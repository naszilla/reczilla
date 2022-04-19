# reczilla

## Install Auto Surprise
```bash
cd Auto-Surprise
$ sudo python setup.py install
```
For more information and instructions, see the [Auto-Surprise README](Auto-Surprise/README.md).


## Download datasets

All datasets are automatically downloaded, except `recipes` (since you need a Kaggle account).
Download `recipes` from [here](https://www.kaggle.com/shuyangli94/food-com-recipes-and-user-interactions/version/2?select=RAW_interactions.csv).
Place `RAW_interactions.csv` inside `Auto-Surprise/sandbox/data/recipes`.

## Run rec sys algorithms

Specify the dataset inside [`experiment.py`](Auto-Surprise/sandbox/experiment.py)
```bash
cd Auto-Surprise/sandbox
python experiment.py
```

## Add more datasets
Look at the methods in [`data_handler.py`](Auto-Surprise/sandbox/data_handler.py), e.g., `get_dating()`, `get_recipes()`. Choose a dataset from our list: https://docs.google.com/spreadsheets/d/1c36DOxVqbMwFe0Wnnoo6lCwH_FUEo6flJtuEGtmeoms/edit#gid=0. Add it to [`data_handler.py`](Auto-Surprise/sandbox/data_handler.py). If you have any questions, ask xxxxx.

Note: for now, we are keeping it pretty simple. We only load the interaction matrix of the dataset, which consists of three rows: `user`, `item`, and `rating`. Later on in the project, we will add user features, item features, and more interaction data such as temporal data.

## Add more algorithms
Our plan is to at least add the algorithms from this paper: https://arxiv.org/abs/1907.06902.

New algorithms should be added using the Surprise base class, and they should have a set of legal hyperparameters defined somewhere. An example algorithm is given in [`custom_algorithms.py`](Auto-Surprise/sandbox/custom_algorithms.py). The hyperparameters here are defined using package hyperopt, which seems like a reasonable choice.

The file [`model_handler.py`](Auto-Surprise/sandbox/model_handler.py) prepares the objects `ALL_ALGORITHMS` and  `ALL_SPACES`.
- `ALL_ALGORITHMS` (dict): each key is an algorithm name and each value is an algorithm (child of surprise.prediction_algorithms.AlgoBase)
- `ALL_SPACES` (dict): each key is an algorithm name, and each value is a parameter space for the corresponding algorithm (a dict consisting of hyperopt.hp objects)

## Run full experiments
Once we have 20 or 30 datasets, and enough algorithms, we will start preliminary experiments and then build an algorithm selector.

The script [`calculate_algorithm_metrics.py`](Auto-Surprise/sandbox/calculate_algorithm_metrics.py) iterates over a list of algorithms and datasets, and calculates some metrics. For each algorithm we sample some random parameter sets using the hyperopt objects defined in [`model_handler.py`](Auto-Surprise/sandbox/model_handler.py).
