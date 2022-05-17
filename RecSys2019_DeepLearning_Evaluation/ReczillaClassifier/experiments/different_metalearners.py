import random
random.seed(7)
from typing import DefaultDict
import numpy as np
import sys
from sklearn.metrics import accuracy_score, precision_score

from ReczillaClassifier.get_alg_feat_selection_data import alg_feature_selection_featurized
from ReczillaClassifier.dataset_families import get_all_datasets, dataset_family_lookup, get_dataset_families
from ReczillaClassifier.dataset_families import family_map as dataset_family_map
from ReczillaClassifier.utils import get_logger, print_special

from ReczillaClassifier.classifier import perc_diff_from_best_subset, perc_diff_from_worst_subset, get_metrics, get_cached_featurized, ALL_DATASET_FAMILIES, METADATASET_NAME

from sklearn.multioutput import RegressorChain
from sklearn.dummy import DummyRegressor
import xgboost as xgb
from tqdm import tqdm

# CONSTANTS
LOGGER = get_logger('different_metalearners')
METRIC = "test_metric_PRECISION_cut_10"
META_LEARNERS = [xgb.XGBRegressor(objective='reg:squarederror'), DummyRegressor()]

# leave one out validation
all_metrics = []

# TODO: iterate over num_algs and num_meta_features, or make these parameters or cli args. pass these to the function
#  alg_feature_selection_featurized
for ml in META_LEARNERS:
    cached_featurized = {}
    for test_dataset_family in tqdm(list(ALL_DATASET_FAMILIES)):
        try:
            test_datasets = list(dataset_family_map.get(test_dataset_family, (test_dataset_family, )))

            print_special(f"{test_dataset_family}, {test_datasets}", LOGGER)
            X_train, y_train, X_test, y_test = get_cached_featurized(METRIC, test_datasets, METADATASET_NAME, cached_featurized)

            # TODO: add baseline methods here: random, knn, other meta-learners, etc.
            base_model = ml
            model = RegressorChain(base_model)
            model.fit(X_train, y_train)
            
            preds = model.predict(X_test)
            metrics = get_metrics(y_test, preds)
            all_metrics.append(metrics)
            print_special(f"{metrics}", LOGGER)
        
        except Exception as e:
            print(f"FAILURE - skipping this dataset family - {test_dataset_family}")
            print(f"EXCEPTION: {e}")
        sys.stdout.flush()

    print_special(f"Metalearner = {ml}", LOGGER)

    accuracies = [m['accuracy'] for m in all_metrics]
    print_special(f"Average leave-one-out accuracy is: {round(100 * np.mean(accuracies), 1)}", LOGGER)

    perc_diff_best = [m['perc_diff_from_best_subset'] for m in all_metrics]
    print_special(f"Average leave-one-out percentage_diff_from_best is: {round(100 * np.mean(perc_diff_best), 1)}", LOGGER)

    perc_diff_worst = [m['perc_diff_from_worst_subset'] for m in all_metrics]
    print_special(f"Average leave-one-out percentage_diff_from_worst is: {round(100 * np.nanmean(perc_diff_worst), 1)}", LOGGER)
