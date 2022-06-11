import random

from ReczillaClassifier.utils import get_logger, print_special
random.seed(7)
from typing import DefaultDict
import numpy as np
import sys
import pickle
from sklearn.metrics import accuracy_score, precision_score

from ReczillaClassifier.get_alg_feat_selection_data import alg_feature_selection_featurized
from ReczillaClassifier.dataset_families import get_all_datasets, dataset_family_lookup, get_dataset_families
from ReczillaClassifier.dataset_families import family_map as dataset_family_map

from ReczillaClassifier.classifier import perc_diff_from_best_subset, perc_diff_from_worst_subset, get_metrics, get_cached_featurized, ALL_DATASET_FAMILIES, METADATASET_NAME, run_metalearner, META_LEARNERS

from sklearn.multioutput import RegressorChain
import xgboost as xgb
from tqdm import tqdm

############### CONSTANTS ###############################
METRIC = "test_metric_PRECISION_cut_10"
LOGGER = get_logger("cunha2018")
ALL_DATASETS = get_all_datasets()
FIXED_ALGS_FEATS = True
NUM_TRIALS = 5
NUM_ALGS = 10
NUM_FEATS = 10

# test_dataset_family = "Movielens"
# test_datasets = list(dataset_family_map.get(test_dataset_family, (test_dataset_family, )))

###############

final_metrics = {}

for test_dataset_family in tqdm(ALL_DATASET_FAMILIES):
    final_metrics[test_dataset_family] = {}
    test_datasets = list(dataset_family_map.get(test_dataset_family, (test_dataset_family, )))

    for exp_id in range(NUM_TRIALS):
        # leave one out validation
        all_metrics = DefaultDict(list)
        cached_featurized = {}
        train_dataset_families = [d for d in ALL_DATASET_FAMILIES if d != test_dataset_family]

        train_datasets = []
        for tdf in train_dataset_families:
            train_datasets += list(dataset_family_map.get(tdf, (tdf, )))

        X_train, y_train, X_test, y_test, y_range_test = get_cached_featurized(
            METRIC, test_datasets, METADATASET_NAME, cached_featurized, train_datasets,
            fixed_algs_feats=FIXED_ALGS_FEATS, num_algs=NUM_ALGS, num_feats=NUM_FEATS, compare_cunha='cunha-2018', random_feats=True)

        for ml in ('svm-poly', ):
            preds = run_metalearner(ml, X_train, y_train, X_test)
            metrics = get_metrics(y_test, y_range_test, preds)
            all_metrics[ml].append(metrics)

        all_metrics['num_algs'].append(NUM_ALGS)
        all_metrics['num_feats'].append(NUM_FEATS)

        final_metrics[test_dataset_family][exp_id] = all_metrics
        print_special(f"{test_dataset_family} \n {all_metrics}", LOGGER)

if FIXED_ALGS_FEATS:
    file_name = "meta_cunha2018_fixed.pkl"
else:
    file_name = "meta_cunha2018.pkl"

pickle.dump(final_metrics, open(f"ReczillaClassifier/results/"+file_name, "wb"))