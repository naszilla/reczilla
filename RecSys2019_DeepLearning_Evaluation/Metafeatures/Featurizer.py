from Data_manager.DataSplitter import DataSplitter
from Data_manager.Dataset import gini_index

from pathlib import Path
import numpy as np
import scipy.stats
import pandas as pd

dataset_folder = "../all_data"

# Get string representation from a feature setting
def feature_string(func_name, func_kwargs):
    if not func_kwargs:
        return func_name
    return "__".join([func_name] + ["{}_{}".format(key, val) for key, val in func_kwargs.items()])

# Entries are (function_name, function_kwargs)
all_features = [
    ("num_users",
     {}),
    ("num_items",
     {}),
    ("num_interactions",
     {}),
    ("sparsity",
     {})
]

# For use with dist_feature. Functions to aggregate distributions.
aggregation_functions = {
    "mean": np.mean,
    "max": np.max,
    "min": np.min,
    "std": np.std,
    "median": np.median,
    "mode": lambda mat: scipy.stats.mode(mat)[0][0],
    #"entropy": scipy.stats.differential_entropy,
    "Gini": gini_index, # Gini index
    "skewness": scipy.stats.skew,
    "kurtosis": scipy.stats.kurtosis,
}

# Compute mean across an axis
def sparse_mean(mat, axis=0):
    return np.array(np.sum(mat, axis=axis)).squeeze() / mat.getnnz(axis=axis)

# For use with dist_feature. Functions to pre-aggregate ratings to form distributions.
pre_aggregation_functions = {
    "mean": sparse_mean,
    "sum": lambda mat, axis: np.array(np.sum(mat, axis=axis)).squeeze(),
    "count": lambda mat, axis: mat.getnnz(axis=axis)
}

# Add all possible dist_features to all_features.
for kind in ["rating", "item", "user"]:
    if kind == "rating":
        pre_agg_funcs = [None]
    else:
        pre_agg_funcs = pre_aggregation_functions.keys()

    for pre_agg_func in pre_agg_funcs:
        for agg_func in aggregation_functions.keys():
            all_features.append((
                "dist_feature",
                {
                    "kind": kind,
                    "pre_agg_func": pre_agg_func,
                    "agg_func": agg_func,
                }
            ))

# Number of users
def num_users(train_set):
    return train_set.shape[0]

# Number of items
def num_items(train_set):
    return train_set.shape[1]

# Number of interactions
def num_interactions(train_set):
    return train_set.nnz

# Sparsity of interaction matrix
def sparsity(train_set):
    return 1 - train_set.nnz / (train_set.shape[0]*train_set.shape[1])

# Distribution feature
def dist_feature(train_set, kind, agg_func, pre_agg_func=None):
    if kind == "rating":
        if pre_agg_func:
            raise RuntimeError("Cannot use pre aggregation function when using all ratings")
        distribution = train_set.data
    else:
        if not pre_agg_func:
            raise RuntimeError("Need pre aggregation function when pre aggregating across items or users")
        if kind == "item":
            distribution = np.array(pre_aggregation_functions[pre_agg_func](train_set, axis=0)).squeeze()
        elif kind == "user":
            distribution = np.array(pre_aggregation_functions[pre_agg_func](train_set, axis=1)).squeeze()
        else:
            raise RuntimeError("Unrecognized kind: {}".format(kind))

    distribution = distribution[~np.isnan(distribution)]
    return aggregation_functions[agg_func](distribution)


def featurize_train_set(train_set, feature_list=None):
    if not feature_list:
        feature_list = all_features

    feature_vals = {}
    for func_name, func_kwargs in feature_list:
        feature_name = feature_string(func_name, func_kwargs)
        feature_vals[feature_name] = globals()[func_name](train_set, **func_kwargs)

    return feature_vals

def featurize_dataset_split(dataset_split_path, feature_list=None):
    dataReader_object, splitter_class, init_kwargs = DataSplitter.load_data_reader_splitter_class(dataset_split_path)
    init_kwargs["forbid_new_split"] = True
    splitter = splitter_class(
        dataReader_object, folder=str(dataset_split_path), verbose=True,
        **init_kwargs
    )
    splitter.load_data()
    train_set = splitter.SPLIT_URM_DICT["URM_train"]
    feature_vals = {
        "dataset_name": dataReader_object.__class__.__name__,
        "split_name": splitter_class.__name__,
    }
    feature_vals.update(featurize_train_set(train_set, feature_list=feature_list))
    return feature_vals


def featurize_all_datasets(folder=dataset_folder):
    path = Path(folder)
    error_paths = []
    dataset_features = []
    for path_match in path.glob("**/data_reader_splitter_class"):
        print(path_match)
        try:
            dataset_features.append(featurize_dataset_split(path_match.parent))
        except Exception as e:
            print(e)
            error_paths.append(path_match.parent)

    dataset_features = pd.DataFrame(dataset_features)
    prefix = ["dataset_name", "split_name"]
    dataset_features = dataset_features[prefix + [col for col in dataset_features if col not in prefix]]
    dataset_features.to_csv("Metafeatures.csv", index=False)

    print("{} datasets not processed.".format(len(error_paths)))
    print([entry.parent.name for entry in error_paths])

featurize_all_datasets()