from Data_manager.Dataset import gini_index
from Metafeatures.utils import register_func

import numpy as np
import scipy.stats

feature_func_lookup = {}

# Returns the entropy of samples in an array. Modelled as discrete distribution.
def sample_entropy(array):
    _, counts = np.unique(array, return_counts=True)
    return scipy.stats.entropy(counts, base=2)

# Compute mean across an axis
def sparse_mean(mat, axis=0):
    return np.array(np.sum(mat, axis=axis)).squeeze() / mat.getnnz(axis=axis)


# For use with dist_feature. Functions to aggregate distributions.
aggregation_functions = {
    "mean": np.mean,
    "max": np.max,
    "min": np.min,
    "std": np.std,
    "median": np.median,
    "mode": lambda mat: scipy.stats.mode(mat)[0][0],
    #"entropy": scipy.stats.differential_entropy,
    "entropy": sample_entropy,
    "Gini": gini_index, # Gini index
    "skewness": scipy.stats.skew,
    "kurtosis": scipy.stats.kurtosis,
}

# For use with dist_feature. Functions to pre-aggregate ratings to form distributions.
pre_aggregation_functions = {
    "mean": sparse_mean,
    "sum": lambda mat, axis: np.array(np.sum(mat, axis=axis)).squeeze(),
    "count": lambda mat, axis: mat.getnnz(axis=axis)
}

# Add all possible dist_features to feature_list.
feature_list = []
for kind in ["rating", "item", "user"]:
    if kind == "rating":
        pre_agg_funcs = [None]
    else:
        pre_agg_funcs = pre_aggregation_functions.keys()

    for pre_agg_func in pre_agg_funcs:
        for agg_func in aggregation_functions.keys():
            feature_list.append((
                "dist_feature",
                {
                    "kind": kind,
                    "pre_agg_func": pre_agg_func,
                    "agg_func": agg_func,
                }
            ))


# Distribution feature
@register_func(feature_func_lookup)
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
    return {"": aggregation_functions[agg_func](distribution)}
