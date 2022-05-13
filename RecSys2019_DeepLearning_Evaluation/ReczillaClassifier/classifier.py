import numpy as np
from sklearn.metrics import accuracy_score, precision_score

from ReczillaClassifier.get_alg_feat_selection_data import alg_feature_selection_featurized
from ReczillaClassifier.dataset_families import get_all_datasets

from sklearn.multioutput import RegressorChain
import xgboost as xgb

ALL_DATASETS = get_all_datasets()

METADATASET_NAME = "metadata-v1"

METRICS = ["test_metric_PRECISION_cut_10", "test_metric_RECALL_cut_10"]

def perc_diff_from_best(labels, outputs, y_test, preds):
    diff = []
    for label, output, label_score, output_score in zip(labels, outputs, y_test, preds):
        m = abs(label_score[label] - label_score[output]) / label_score[label]
        diff.append(m)
    return np.mean(diff)

def perc_diff_from_worst(labels, outputs, y_test, preds):
    diff = []
    for label, output, label_score, output_score in zip(labels, outputs, y_test, preds):
        m = abs(label_score[output] - min(label_score)) / (label_score[label] - min(label_score))
        diff.append(m)
    return np.mean(diff)

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

# leave one out validation
all_metrics = []

# TODO: iterate over num_algs and num_meta_features, or make these parameters or cli args. pass these to the function
#  alg_feature_selection_featurized
for _metric in METRICS:
    for test_dataset in list(ALL_DATASETS):
        X_train, y_train, X_test, y_test = alg_feature_selection_featurized(_metric, [test_dataset], METADATASET_NAME)

        # TODO: add baseline methods here: random, knn, other meta-learners, etc.
        base_model = xgb.XGBRegressor(objective='reg:squarederror')
        model = RegressorChain(base_model)
        model.fit(X_train, y_train)
        
        preds = model.predict(X_test)
        metrics = get_metrics(y_test, preds)
        all_metrics.append(metrics)

    # import pdb; pdb.set_trace()
    print("Metric = ", _metric)

    accuracies = [m['accuracy'] for m in all_metrics]
    print("Average leave-one-out accuracy is: ", round(100 * np.mean(accuracies), 1))

    perc_diff_best = [m['perc_diff_from_best'] for m in all_metrics]
    print("Average leave-one-out percentage_diff_from_best is: ", round(100 * np.mean(perc_diff_best), 1))

    perc_diff_worst = [m['perc_diff_from_worst'] for m in all_metrics]
    print("Average leave-one-out percentage_diff_from_worst is: ", round(100 * np.nanmean(perc_diff_worst), 1))