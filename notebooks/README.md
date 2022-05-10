### prepare_metadataset.ipynb

This notebook creates a meta-dataset for all RecZilla experiments, using aggregated output from all experiments. The only input to this notebook is a CSV produced by the script `process_inbox.py` -- this script reads all zipped results from gcloud and creates an aggregated CSV.

### meta_datasets

The meta-datasets can be downloaded from google drive. These should be read as a pickle file (using `pandas.read_pickle(<filename>)`):
- v0 meta-dataset: this includes only the results from our workshop submission: <tbd>
- v1: this includes all results that have completed by May 9: <tbd> 
  
In these datasets, there is one row for each combination of algorhtm + hyperparameter set + dataset split. The important columns are:
- `alg_family`: the algorithm family (e.g. TopPop or UserKNN)
- `hyperparameters_source`: the method used to create the hyperparameters used. This string ends with either `default`, `random_i` (i = an index to identify unique sets of random hyperparameters). Strings that start with a similarity measure (`euclidean` or `assymetric`) are specific to UserKNN or ItemKNN.
- `alg_param_name`: this column uniquely describes the (algorithm + hyperparameters) used, combined with a `:`. For example, `UserKNNCF:asymmetric_random_2`.
- columns that start with `test_metric_` or `val_metric_` are metrics calculated on the test set or validation set. These column names end with the cutoff used to calculate them (e.g. `_cut_5`).
- columns that start with `param_` are hyperparameters. These are just for reference, in case we want to look up actual hyperparameter values.
