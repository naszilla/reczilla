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
