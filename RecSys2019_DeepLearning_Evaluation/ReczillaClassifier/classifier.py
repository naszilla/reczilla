import random
import numpy as np
import sys
from sklearn.metrics import accuracy_score, precision_score
from sklearn.multioutput import RegressorChain, MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import xgboost as xgb

from ReczillaClassifier.get_alg_feat_selection_data import alg_feature_selection_featurized
from ReczillaClassifier.dataset_families import get_all_datasets, dataset_family_lookup, get_dataset_families
from ReczillaClassifier.dataset_families import family_map as dataset_family_map

ALL_DATASET_FAMILIES = sorted(get_dataset_families() - set(['GoogleLocalReviewsReader']))

METADATASET_NAME = "metadata-v2"

def run_metalearner(model_name, X_train, y_train, X_test):
    """

    Args:
        model_name: "xgboost", "knn"
        X_train: training data
        y_train: training target
        X_test: testing data

    Returns:
        preds: predictions
    """
    if model_name == "xgboost":
        base_model = xgb.XGBRegressor(objective='reg:squarederror')
        model = RegressorChain(base_model)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

    elif model_name == "knn":
        n_neighbors = 5
        n_training_samples = X_train.shape[0]
        pipe = Pipeline([("scaler", StandardScaler()),
                         ("knn", KNeighborsRegressor(n_neighbors=min(n_neighbors, n_training_samples)))])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

    elif model_name == "linear":
        pipe = Pipeline([("scaler", StandardScaler()),
                         ("linear", MultiOutputRegressor(Ridge(alpha=10))),
                          ])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

    elif model_name == "random":
        num_algs = len(y_train[0])
        preds = []
        for x_t in X_test:
            one_hot = [0] * num_algs
            one_hot[random.randint(0, num_algs-1)] = 1
            preds.append(one_hot)
    else:
        raise NotImplementedError("{} not implemented".format(model_name))

    return preds

def perc_diff_from_best_global(outputs, y_test, y_range_test):
    diff = []
    for output, label_score, score_range in zip(outputs, y_test, y_range_test):
        best, worst = score_range
        m = abs(best - label_score[output]) / (best - worst)
        diff.append(m)
    return np.nanmean(diff)

def perc_diff_from_worst_global(outputs, y_test, y_range_test):
    diff = []
    for output, label_score, score_range in zip(outputs, y_test, y_range_test):
        best, worst = score_range
        m = abs(label_score[output] - worst) / (best - worst)
        diff.append(m)
    return np.nanmean(diff)

def perc_diff_from_best_subset(labels, outputs, y_test, preds):
    diff = []
    for label, output, label_score, output_score in zip(labels, outputs, y_test, preds):
        m = abs(label_score[label] - label_score[output]) / (label_score[label] - min(label_score))
        diff.append(m)
    return np.nanmean(diff)

def perc_diff_from_worst_subset(labels, outputs, y_test, preds):
    diff = []
    for label, output, label_score, output_score in zip(labels, outputs, y_test, preds):
        m = abs(label_score[output] - min(label_score)) / (label_score[label] - min(label_score))
        diff.append(m)
    return np.nanmean(diff)

def get_metrics(y_test, y_range_test, preds):
    metrics = {}
    labels = [np.argmax(yt) for yt in y_test]
    outputs = [np.argmax(p) for p in preds]

    # TODO: does this collect the two accuracy metrics described in the paper? ("%OPT" and "AlgAccuracy"). we might need
    #  to add some logic to check if the selected algorithm matches the ground-truth-best algorithm.
    # metrics['precision'] = np.mean(precision_score(labels, outputs, average=None))
    metrics['accuracy'] = accuracy_score(labels, outputs)
    metrics["perc_diff_from_best_global"] = perc_diff_from_best_global(outputs, y_test, y_range_test)
    metrics["perc_diff_from_worst_global"] = perc_diff_from_worst_global(outputs, y_test, y_range_test)
    metrics['perc_diff_from_best_subset'] = perc_diff_from_best_subset(labels, outputs, y_test, preds)
    metrics['perc_diff_from_worst_subset'] = perc_diff_from_worst_subset(labels, outputs, y_test, preds)
    return metrics

def get_cached_featurized(metric_name, test_datasets, dataset_name, cached_featurized = {}, train_datasets=None, fixed_algs_feats=False):
    test_families = tuple(sorted(set(dataset_family_lookup(test_dataset) for test_dataset in test_datasets)))
    if test_families not in cached_featurized or train_datasets is not None:
        cached_featurized[test_families] = alg_feature_selection_featurized(metric_name, test_datasets, dataset_name, train_datasets, fixed_algs_feats=fixed_algs_feats)
    return cached_featurized[test_families]

if __name__ == "__main__":
    X_train, y_train, X_test, y_test, y_best_test = get_cached_featurized("test_metric_PRECISION_cut_10", ["AnimeReader"], METADATASET_NAME)

    preds = run_metalearner("logreg", X_train, y_train, X_test)
    metrics = get_metrics(y_test, y_best_test, preds)

    print(f"metrics: {metrics}")