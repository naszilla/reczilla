import numpy as np
import sys
from sklearn.metrics import accuracy_score, precision_score

from ReczillaClassifier.get_alg_feat_selection_data import alg_feature_selection_featurized
from ReczillaClassifier.dataset_families import get_all_datasets, dataset_family_lookup, get_dataset_families
from ReczillaClassifier.dataset_families import family_map as dataset_family_map

ALL_DATASET_FAMILIES = sorted(get_dataset_families())

METADATASET_NAME = "metadata-v1.1"

def perc_diff_from_best(labels, outputs, y_test, preds):
    diff = []
    for label, output, label_score, output_score in zip(labels, outputs, y_test, preds):
        m = abs(label_score[label] - label_score[output]) / (label_score[label] - min(label_score))
        diff.append(m)
    return np.nanmean(diff)

def perc_diff_from_worst(labels, outputs, y_test, preds):
    diff = []
    for label, output, label_score, output_score in zip(labels, outputs, y_test, preds):
        m = abs(label_score[output] - min(label_score)) / (label_score[label] - min(label_score))
        diff.append(m)
    return np.nanmean(diff)

def get_metrics(y_test, preds):
    metrics = {}
    labels = [np.argmax(yt) for yt in y_test]
    outputs = [np.argmax(p) for p in preds]

    # TODO: does this collect the two accuracy metrics described in the paper? ("%OPT" and "AlgAccuracy"). we might need
    #  to add some logic to check if the selected algorithm matches the ground-truth-best algorithm.
    # metrics['precision'] = np.mean(precision_score(labels, outputs, average=None))
    metrics['accuracy'] = accuracy_score(labels, outputs)
    metrics['perc_diff_from_best'] = perc_diff_from_best(labels, outputs, y_test, preds)
    metrics['perc_diff_from_worst'] = perc_diff_from_worst(labels, outputs, y_test, preds)
    return metrics

def get_cached_featurized(metric_name, test_datasets, dataset_name, cached_featurized = {}, train_datasets=None):
    test_families = tuple(sorted(set(dataset_family_lookup(test_dataset) for test_dataset in test_datasets)))
    if test_families not in cached_featurized or train_datasets is not None:
        cached_featurized[test_families] = alg_feature_selection_featurized(metric_name, test_datasets, dataset_name, train_datasets)
    return cached_featurized[test_families]