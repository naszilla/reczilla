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
Look at the methods in [`data_handler.py`](Auto-Surprise/sandbox/data_handler.py), e.g., `get_dating()`, `get_recipes()`. Choose a dataset from our list: https://docs.google.com/spreadsheets/d/1c36DOxVqbMwFe0Wnnoo6lCwH_FUEo6flJtuEGtmeoms/edit#gid=0. Add it to [`data_handler.py`](Auto-Surprise/sandbox/data_handler.py). If you have any questions, ask Sujay.

Note: for now, we are keeping it pretty simple. We only load the interaction matrix of the dataset, which consists of three rows: `user`, `item`, and `rating`. Later on in the project, we will add user features, item features, and more interaction data such as temporal data.

## Add more algorithms
Our plan is to at least add the algorithms from this paper: https://arxiv.org/abs/1907.06902.

## Run full experiments
Once we have 20 or 30 datasets, and enough algorithms, we will start preliminary experiments and then build an algorithm selector.
