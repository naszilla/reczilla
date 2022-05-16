import random
random.seed(7)
from typing import DefaultDict
import numpy as np
import sys
from sklearn.metrics import accuracy_score, precision_score

from ReczillaClassifier.get_alg_feat_selection_data import alg_feature_selection_featurized
from ReczillaClassifier.dataset_families import get_all_datasets, dataset_family_lookup, get_dataset_families
from ReczillaClassifier.dataset_families import family_map as dataset_family_map

from ReczillaClassifier.classifier import perc_diff_from_best, perc_diff_from_worst, get_metrics, get_cached_featurized, ALL_DATASET_FAMILIES, METADATASET_NAME

from sklearn.multioutput import RegressorChain
import xgboost as xgb
from tqdm import tqdm

############### CONSTANTS ###############################
METRIC = "test_metric_PRECISION_cut_10"
ALL_DATASETS = get_all_datasets()

test_dataset_family = "Movielens"
test_datasets = list(dataset_family_map.get(test_dataset_family, (test_dataset_family, )))

###############

final_metrics = DefaultDict(list)

for num_train in tqdm(range(2, 20, 2)):
    try:
        # leave one out validation
        all_metrics = []
        cached_featurized = {}
        for _ in range(5):
            train_dataset_families = [d for d in ALL_DATASET_FAMILIES if d != test_dataset_family]
            train_dataset_families = random.sample(train_dataset_families, k=num_train)
            train_datasets = []
            for tdf in train_dataset_families:
                train_datasets += list(dataset_family_map.get(tdf, (tdf, )))

            X_train, y_train, X_test, y_test = get_cached_featurized(METRIC, test_datasets, METADATASET_NAME, cached_featurized, train_datasets)

            base_model = xgb.XGBRegressor(objective='reg:squarederror')
            model = RegressorChain(base_model)
            model.fit(X_train, y_train)
            
            preds = model.predict(X_test)
            metrics = get_metrics(y_test, preds)
            all_metrics.append(metrics)

        final_metrics['num_train'].append(num_train)

        accuracies = [m['accuracy'] for m in all_metrics]
        final_metrics['accuracy'].append(round(100 * np.mean(accuracies), 1))

        perc_diff_best = [m['perc_diff_from_best'] for m in all_metrics]
        final_metrics['perc_diff_from_best'].append(round(100 * np.mean(perc_diff_best), 1))

        perc_diff_worst = [m['perc_diff_from_worst'] for m in all_metrics]
        final_metrics['perc_diff_from_worst'].append(round(100 * np.nanmean(perc_diff_worst), 1))
    except:
        import pdb; pdb.set_trace()
print(final_metrics)
