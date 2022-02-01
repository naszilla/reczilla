# Installation

Requires python 3.6. Make sure all requirements are installed, with pip. It's a good idea to set up a virtual env with python>=3.6.

```
python -m pip install -r requirements.txt
```

Some of the algorithms require compiling Cython files. Compile these using:

```
python run_compile_all_cython.py
```

# General Info

This code is not perfectly documented, so here is some general info I've gathered from the code and READMEs.

## Algorithms

All algorithms have the same basic interface, the constructor takes as input only sparse matrices of scipy.sparse format.

All recommender take as first argument in the constructor the URM, content-based models also take the ICM or UCM as the second.
- **User Rating Matrix (URM)** of shape |n users|x|n items| containing the user-item interactions, either implicit (1-0) or explicit (any value)
- **Item Content Matrix (ICM)** of shape |n items|x|n item features| containing the item features, again with any numerical value
- **User Content Matrix (UCM)** of shape |n users|x|n users features| containing the item features, again with any numerical value

### Algorithm Parameters

- the training dataset is passed as an argument upon initialization. all other parameters/hyperparameters are passed during the call to `fit()`
- it looks like the `fit()` method should only be called once, because some algorithm instance attribtues are changed by some calls to `fit()`. (This is true at least for `ItemKNNCFRecommender`). 

### Parameter Tuning

- **fixed params:** the class `SearchInputRecommenderArgs` is used to pass *fixed* parameters to the algorithm. E.g. if we want to fix k=5 in KNN, we would set this in an instance of `SearchInputRecommenderArgs`. This instance is usually assigned to the variable `recommender_input_args`, and is passed to the hyperparameter search function. This object is then used (e.g.) by a search class instance. For example, the function `SearchBayesianSkopt.search()` takes two positional arguments: the first is `recommender_input_args` (fixed parameters) and the second is the search space ("variable" parameters). 
- **variable params:** variable parameters (the search space) are passed as a dictionary: each key is the name of a kwarg passed to `fit()`, and each value is a parameter range, and must be an instance of `skopt.space.space.Real`, `skopt.space.space.Integer`, or `skopt.space.space.Categorical`.

For evaluating a particular set of hyperparameters, we can use this function for inspiration:
https://github.com/MaurizioFD/RecSys2019_DeepLearning_Evaluation/blob/0fb6b7f5c396f8525316ed66cf9c9fdb03a5fa9b/ParameterTuning/SearchAbstractClass.py#L271


## Data

...

# New Code

## ParameterTuning.RandomSearch (class)

This class is a random search over a fixed number of hyperparameter samples. All metrics & hyperparameters are saved to the metadata. But we will write a method to save these params/metrics separately.

## test.py

This is a simple driver script, currently just for debugging.

# TODO

- add random seed to data splitter (in files `Data_manager.split_functions`, and places where this code is used.)
- write some code to extract data features using this code's API
- add a class method to `ParameterTuning.RandomSearch` that saves a CSV or json with hyperparameters + train/test/validation metrics.