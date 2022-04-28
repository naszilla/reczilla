import numpy as np
from sklearn.metrics import accuracy_score, precision_score

from ReczillaClassifier.get_alg_feat_selection_data import alg_feature_selection_featurized, ALL_DATASETS

from sklearn.multioutput import RegressorChain
import xgboost as xgb

METRICS = ["PRECISION_cut_10", "MAP_cut_10"]

def get_metrics(y_test, preds):
    metrics = {}
    labels = [np.argmax(yt) for yt in y_test]
    outputs = [np.argmax(p) for p in preds]
    
    # if np.min(y_test[0]) == np.max(y_test[0]):
    #     print(y_test)
    #     exit()

    # metrics['precision'] = np.mean(precision_score(labels, outputs, average=None))
    metrics['accuracy'] = accuracy_score(labels, outputs)
    metrics['perc_diff_from_best'] = np.mean([abs(l_s[l] - l_s[o])/l_s[l]  for l, o, l_s, o_s  in zip(labels, outputs, y_test, preds)])
    metrics['perc_diff_from_worst'] = np.mean([abs(l_s[o] - np.min(l_s))/(l_s[l] - np.min(l_s))  for l, o, l_s, o_s  in zip(labels, outputs, y_test, preds)])

    return metrics

# leave one out validation
all_metrics = []

for _metric in METRICS:
    for test_dataset in ALL_DATASETS:
        X_train, y_train, X_test, y_test = alg_feature_selection_featurized(_metric, [test_dataset])

        base_model = xgb.XGBRegressor(objective='reg:squarederror')
        model = RegressorChain(base_model)
        model.fit(X_train, y_train)
        
        preds = model.predict(X_test)
        metrics = get_metrics(y_test, preds)
        all_metrics.append(metrics)

    print("Metric = ", _metric)

    accuracies = [m['accuracy'] for m in all_metrics]
    print("Average leave-one-out accuracy is: ", round(100 * np.mean(accuracies), 1))

    perc_diff_best = [m['perc_diff_from_best'] for m in all_metrics]
    print("Average leave-one-out percentage_diff_from_best is: ", round(100 * np.mean(perc_diff_best), 1))

    perc_diff_worst = [m['perc_diff_from_worst'] for m in all_metrics]
    print("Average leave-one-out percentage_diff_from_worst is: ", round(100 * np.nanmean(perc_diff_worst), 1))