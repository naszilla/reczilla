# Installation

Requires python 3.6. Make sure all requirements are installed, with pip. It's a good idea to set up a virtual env with python>=3.6.

```
python -m pip install -r requirements.txt
```

Some of the algorithms require compiling Cython files. Compile these using:

```
python run_compile_all_cython.py
```

# Unittests & New Algs/Datasets

## Datasets

We keep track of all datasets using file `RecSys2019_DeepLearning_Evaluation/dataset_handler.py`. All datasets are stored in the list `DATASET_READER_LIST`. Datasets should be retrieved by name, using function `dataset_handler.dataset_handler()`.

**To add a new dataset**, do the following:
1. Add an import statement for the datareader to file `dataset_handler.py`. The datareader must be a subclass of `Data_manager.DataReader`.
2. Add the datareader object to the list `dataset_handler.DATASET_READER_LIST`.

## Algorithms

We keep track of all algorithms using the file `RecSys2019_DeepLearning_Evaluation/algorithm_handler.py`. The algorithm handler is a bit more complicated than the dataset handler, because we need to specify parameter spaces and early stopping params for each algorithm. (**NOTE:** to clean this up, we can make parameter spaces attributes of the algorithm classes.)

**To add a new algorithm**, do the following:
1. Add an import statement for the algorithm to the file `algorithm_handler.py`. The algorithm must be a subclass of `Base.BaseRecommender` (or a subclass of one its many subclasses, such as `BaseSimilarityMatrixRecommender`).
2. Add the algorithm class name to list `algorithm_handler.ALGORITHM_NAME_LIST`.
3. To add a hyperparameter space, add an `elif` clause to function `algorithm_handler.algorithm_handler`. Define the hyperparameter space in dictionary `space`, and add any kwargs that should be passed to the algorithm's `fit()` function to dict `fit_keyword_args`.
4. Test that the algorithm works by running `tests.algorithm_test`. This will test all algorithms listed in `algorithm_handler.ALGORITHM_NAME_LIST`.

# Codebase

The codebase we are building on is not perfectly documented, so here is some general info I've gathered from the code and READMEs.

## Algorithms

All algorithms have the same basic interface, the constructor takes as input only sparse matrices of scipy.sparse format.

All recommender take as first argument in the constructor the URM, content-based models also take the ICM or UCM as the second.
- **User Rating Matrix (URM)** of shape |n users|x|n items| containing the user-item interactions, either implicit (1-0) or explicit (any value)
- **Item Content Matrix (ICM)** of shape |n items|x|n item features| containing the item features, again with any numerical value
- **User Content Matrix (UCM)** of shape |n users|x|n users features| containing the item features, again with any numerical value
- all matrices appear to be in CSR format (https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html)
- assuming missing values are 0s

### Algorithm Parameters

- the training dataset is passed as an argument upon initialization. all other parameters/hyperparameters are passed during the call to `fit()`
- it looks like the `fit()` method should only be called once, because some algorithm instance attribtues are changed by some calls to `fit()`. (This is true at least for `ItemKNNCFRecommender`). 

### Parameter Tuning

- **fixed params:** the class `SearchInputRecommenderArgs` is used to pass *fixed* parameters to the algorithm. E.g. if we want to fix k=5 in KNN, we would set this in an instance of `SearchInputRecommenderArgs`. This instance is usually assigned to the variable `recommender_input_args`, and is passed to the hyperparameter search function. This object is then used (e.g.) by a search class instance. For example, the function `SearchBayesianSkopt.search()` takes two positional arguments: the first is `recommender_input_args` (fixed parameters) and the second is the search space ("variable" parameters). 
- **variable params:** variable parameters (the search space) are passed as a dictionary: each key is the name of a kwarg passed to `fit()`, and each value is a parameter range, and must be an instance of `skopt.space.space.Real`, `skopt.space.space.Integer`, or `skopt.space.space.Categorical`.

For evaluating a particular set of hyperparameters, we can use this function for inspiration:
https://github.com/MaurizioFD/RecSys2019_DeepLearning_Evaluation/blob/0fb6b7f5c396f8525316ed66cf9c9fdb03a5fa9b/ParameterTuning/SearchAbstractClass.py#L271

## Evaluation

All model evaluations are carried out by a subclass of the `Evaluation.Evaluator` class. `Evaluator` is the base class, and can't actually be used for evaluation. The specific evaluation is carried out by class function `_run_evaluation_on_selected_users`, defined for each of the subclasses.

There appear to be two evaluators defined:
- `Evaluation.Evaluator.EvaluatorHoldout`: not sure exactly what this does.
- `Evaluation.Evaluator.EvaluatorNegativeItemSample`: not sure exactly what this does. 
## Data

### `DataReader`
- each dataset needs to have a reader class, which is a subclass of `DataReader`. 
- each dataset is loaded using the class function `load_data()`
- loaded datasets are saved to disk, and the location is specified by the kwarg `save_folder_path`. This is the only argument for `load_data()`

### `DataSplitter`
- datasplitters are initialized with a `DataReader` object/subclass, and function in a very similar way
- when a datasplitter is initiated, it saves the full dataset locally. different subfolders are used for each split
- splits are created using the `load_data()` function, which takes a single kwarg `save_folder_path`. This function tries to find an existing dataset split in `save_folder_path`. If no data is found, it creates a new split and loads it.

`DataSplitter` subclasses:
- `DataSplitter_leave_k_out`: create a test set that contains k holdout interactions for each user. the validation set contains k*2 interactions (I think..), and the train set contains all remaining interactions.

# New Code

## `ParameterTuning.RandomSearch (class)`

This class is a random search over a fixed number of hyperparameter samples. All metrics & hyperparameters are saved to the metadata. But we will write a method to save these params/metrics separately.

## `algorithm_handler`

This script contains a single function `algorithm_handler()`, which takes an algorithm name as input and returns:
- the algorithm class
- a hyperparameter space for the algorithm
- a `SearchInputRecommenderArgs` object, which contains params always passed to the alg init and fit functions.

The set of algorithms that can be passed to `algorithm_handler()` are in `ALGORITHM_NAME_LIST`.



# TODO

- add DL algs to algorithm_handler
- add random seed to data splitter (in files `Data_manager.split_functions`, and places where this code is used.)
- save train metrics to metadata for each sample, rather than for only one. this will be helpful for checking overfitting
- write some code to extract features using this code's API. would be good to use the existing dataloader as an interface (e.g. dataSplitter.load_data). may be good to write a new class for this ("Dataset"?)
- there are some datasets in subfolders (e.g. `RecSys2019_DeepLearning_Evaluation/Conferences/IJCAI/DMF_our_interface/AmazonMusicReader/AmazonMusicReader.py`). check whether we want to include any of these in our data handler.
- (maybe) prevent dataIO.save_data() to overwrite certain files. we can also just check this manually. since metadata is rewritten during each call to _objective_function(), it might save time to prevent this rewriting
- (maybe) make hyperparameter spaces class arttributes for each algorithm, rather than specifying them separately in `algorithm_handler.py`
