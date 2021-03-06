# Installation

Requires python 3.6. Make sure all requirements are installed, with pip. It's a good idea to set up a virtual env with python>=3.6.

```
python -m pip install -r requirements.txt
```

Some of the algorithms require compiling Cython files. Compile these using:

```
python run_compile_all_cython.py
```

# Data

## Preparing Datasets

Before using recsys datasets with this codebase, we need to build the following obejcts for each:
- Dataset attributes (metadata)
- **User Rating Matrix (URM)** of shape |n users|x|n items| containing the user-item interactions, either implicit (1-0) or explicit (any value)
- [optional] **Item Content Matrix (ICM)** of shape |n items|x|n item features| containing the item features, again with any numerical value
- [optional] **User Content Matrix (UCM)** of shape |n users|x|n users features| containing the item features, again with any numerical value

Each of these items are built by the dataset's DataReader object (`Data_manager.DataReader.DataReader`). For each dataset, the DataReader object reads the original data (by downloading it, or looking for a file locally), and then constructing the metadata and URM, and the ICM and UCM if they are available.

The metadata/URM/UCM/ICM are then written as zip files to a directory specified by the DataReader function `load_data()`. The script `Data_manager/download_check_all_data.py` downloads all datasets defined in `dataset_handler.py`, and runs a basic sanity check on them (see next section).

The files that are written by each dataloader can include:
- `dataset_global_attributes.zip`
- `dataset_URM.zip` 
- [optional] `dataset_ICM_mappers.zip`
- [optional] `dataset_UCM_mappers.zip`
- [optional] `dataset_additional_mappers.zip`

## Keeping Track of Datasets

We keep track of all datasets using script `dataset_handler.py`. All datasets are stored in the list `DATASET_READER_LIST`. Datasets should be retrieved by name, using function `dataset_handler.dataset_handler()`.

**To add a new dataset**, do the following:
1. Add an import statement for the datareader to file `dataset_handler.py`. The datareader must be a subclass of `Data_manager.DataReader`.
2. Add the datareader object to the list `dataset_handler.DATASET_READER_LIST`.

## Reading Prepared Datasets

After a dataset has been loaded and saved using its DataReader object (with function `DataReader.load_data()`), then we can easily read the prepared data using a Dataset object (`Data_manager.Dataset.Dataset`). The function `Dataset.load_data` will attempt to read each of the zip files written by the DataReader.

# Algorithms

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

### Data Splitting Functions

The datasplitter objects may be more complicated than we need. Instead, we might just use the splitting functions in `Data_manager.split_functions`. The functions in each of these files take a URM (sparse matrix) as input, and return a train/test/validation split (up to three other sparse matrices).

### Storing Original Data

Original datasets are downloaded and stored to `./Data_manager_split_datasets`. This path is hard-coded as attribute `DATASET_SPLIT_ROOT_FOLDER` in `Data_manager/DataReader.py`.

This original data is usually downloaded when we call `load_data()` on a datareader object, which in turn calls the hidden function `_load_from_original_file()`. This function is defined differently for each datareader.

- **TODO:** (maybe) change this so that original data is downloaded to a directory of our choice, rather than `./Data_manager_split_datasets`.

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

- add random seed to all algorithms that use randomness
- add random seed to data splitter (in files `Data_manager.split_functions`, and places where this code is used.)
- save train metrics to metadata for each sample, rather than for only one. this will be helpful for checking overfitting
- write some code to extract features using this code's API. would be good to use the existing dataloader as an interface (e.g. dataSplitter.load_data). may be good to write a new class for this ("Dataset"?)
- there are some datasets in subfolders (e.g. `RecSys2019_DeepLearning_Evaluation/Conferences/IJCAI/DMF_our_interface/AmazonMusicReader/AmazonMusicReader.py`). check whether we want to include any of these in our data handler.
- (maybe) make hyperparameter spaces class arttributes for each algorithm, rather than specifying them separately in `algorithm_handler.py`