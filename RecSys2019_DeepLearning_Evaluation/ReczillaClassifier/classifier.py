import numpy as np
import sys
from sklearn.metrics import accuracy_score, precision_score

from ReczillaClassifier.get_alg_feat_selection_data import alg_feature_selection_featurized
from ReczillaClassifier.dataset_families import get_all_datasets, dataset_family_lookup, get_dataset_families
from ReczillaClassifier.dataset_families import family_map as dataset_family_map
from ReczillaClassifier.utils import print_special

from sklearn.multioutput import RegressorChain
import xgboost as xgb
from tqdm import tqdm

ALL_DATASET_FAMILIES = sorted(get_dataset_families())

METADATASET_NAME = "metadata-v1"

METRICS = ["test_metric_PRECISION_cut_10", "test_metric_RECALL_cut_10"]

def perc_diff_from_best(labels, outputs, y_test, preds):
    diff = []
    for label, output, label_score, output_score in zip(labels, outputs, y_test, preds):
        m = abs(label_score[label] - label_score[output]) / label_score[label]
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
    
    # if np.min(y_test[0]) == np.max(y_test[0]):
    #     print(y_test)
    #     exit()

    # TODO: does this collect the two accuracy metrics described in the paper? ("%OPT" and "AlgAccuracy"). we might need
    #  to add some logic to check if the selected algorithm matches the ground-truth-best algorithm.
    # metrics['precision'] = np.mean(precision_score(labels, outputs, average=None))
    metrics['accuracy'] = accuracy_score(labels, outputs)
    metrics['perc_diff_from_best'] = perc_diff_from_best(labels, outputs, y_test, preds)
    metrics['perc_diff_from_worst'] = perc_diff_from_worst(labels, outputs, y_test, preds)
    return metrics

def get_cached_featurized(metric_name, test_datasets, dataset_name, cached_featurized = {}):
    test_families = tuple(sorted(set(dataset_family_lookup(test_dataset) for test_dataset in test_datasets)))
    if test_families not in cached_featurized:
        cached_featurized[test_families] = alg_feature_selection_featurized(_metric, test_datasets, METADATASET_NAME)
    return cached_featurized[test_families]


# leave one out validation
all_metrics = []

# TODO: iterate over num_algs and num_meta_features, or make these parameters or cli args. pass these to the function
#  alg_feature_selection_featurized
for _metric in METRICS:
    cached_featurized = {}
    for test_dataset_family in tqdm(list(ALL_DATASET_FAMILIES)):
        try:
            test_datasets = list(dataset_family_map.get(test_dataset_family, (test_dataset_family, )))

            print_special(f"{test_dataset_family}, {test_datasets}")
            X_train, y_train, X_test, y_test = get_cached_featurized(_metric, test_datasets, METADATASET_NAME, cached_featurized)

            # TODO: add baseline methods here: random, knn, other meta-learners, etc.
            base_model = xgb.XGBRegressor(objective='reg:squarederror')
            model = RegressorChain(base_model)
            model.fit(X_train, y_train)
            
            preds = model.predict(X_test)
            metrics = get_metrics(y_test, preds)
            all_metrics.append(metrics)
            print_special(f"{metrics}")
        
        except Exception as e:
            print(f"FAILURE - skipping this dataset family - {test_dataset_family}")
            print(f"EXCEPTION: {e}")
        sys.stdout.flush()

    print_special("Metric = ", _metric)

    accuracies = [m['accuracy'] for m in all_metrics]
    print_special("Average leave-one-out accuracy is: ", round(100 * np.mean(accuracies), 1))

    perc_diff_best = [m['perc_diff_from_best'] for m in all_metrics]
    print_special("Average leave-one-out percentage_diff_from_best is: ", round(100 * np.mean(perc_diff_best), 1))

    perc_diff_worst = [m['perc_diff_from_worst'] for m in all_metrics]
    print_special("Average leave-one-out percentage_diff_from_worst is: ", round(100 * np.nanmean(perc_diff_worst), 1))