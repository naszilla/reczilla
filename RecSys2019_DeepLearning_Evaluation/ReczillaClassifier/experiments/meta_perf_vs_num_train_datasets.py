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

from ReczillaClassifier.classifier import perc_diff_from_best, perc_diff_from_worst, get_metrics, get_cached_featurized, ALL_DATASET_FAMILIES, METADATASET_NAME, run_metalearner

from sklearn.multioutput import RegressorChain
import xgboost as xgb
from tqdm import tqdm

############### CONSTANTS ###############################
METRIC = "test_metric_PRECISION_cut_10"
LOGGER = get_logger("meta_perf_vs_num_train_datasets")
ALL_DATASETS = get_all_datasets()
META_LEARNERS = ['xgboost', 'random', 'knn']
FIXED_ALGS_FEATS = False

# test_dataset_family = "Movielens"
# test_datasets = list(dataset_family_map.get(test_dataset_family, (test_dataset_family, )))

###############

final_metrics = {}

for test_dataset_family in tqdm(ALL_DATASET_FAMILIES):
    final_metrics[test_dataset_family] = {}
    test_datasets = list(dataset_family_map.get(test_dataset_family, (test_dataset_family, )))

    for exp_id in range(5):
        # leave one out validation
        all_metrics = DefaultDict(list)
        cached_featurized = {}
        train_dataset_families = [d for d in ALL_DATASET_FAMILIES if d != test_dataset_family]
        random.shuffle(train_dataset_families)
        jump = 2
        train_datasets = []

        for num_train in range(jump, 20, jump):
            for tdf in train_dataset_families[num_train - jump:num_train]:
                train_datasets += list(dataset_family_map.get(tdf, (tdf, )))

            X_train, y_train, X_test, y_test = get_cached_featurized(METRIC, test_datasets, METADATASET_NAME, cached_featurized, train_datasets, fixed_algs_feats=FIXED_ALGS_FEATS)

            for ml in META_LEARNERS:
                preds = run_metalearner(ml, X_train, y_train, X_test)
                metrics = get_metrics(y_test, preds)
                all_metrics[ml].append(metrics)
            
            all_metrics['num_train'].append(num_train)

        final_metrics[test_dataset_family][exp_id] = all_metrics
        print_special(f"{test_dataset_family} \n {all_metrics}", LOGGER)


    # print_special("ALL RESULTS: \n", LOGGER)

    # print_special(f"{final_metrics}", LOGGER)
pickle.dump(final_metrics, open("ReczillaClassifier/results/meta_perf_vs_num_train.pkl", "wb"))