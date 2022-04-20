import numpy as np
from sklearn.metrics import accuracy_score, precision_score

from ReczillaClassifier.get_alg_feat_selection_data import alg_feature_selection_featurized, ALL_DATASETS

from sklearn.multioutput import RegressorChain
import xgboost as xgb

METRIC = "PRECISION_cut_1"

def get_metrics(y_test, preds):
    metrics = {}
    labels = [np.argmax(yt) for yt in y_test]
    outputs = [np.argmax(p) for p in preds]

    # metrics['precision'] = np.mean(precision_score(labels, outputs, average=None))
    metrics['accuracy'] = accuracy_score(labels, outputs)
    metrics['perc_diff_from_best'] = np.mean([abs(l_s[l] - l_s[o])/l_s[l]  for l, o, l_s, o_s  in zip(labels, outputs, y_test, preds)])

    return metrics

# leave one out validation
all_metrics = []

for test_dataset in ALL_DATASETS:
    X_train, y_train, X_test, y_test = alg_feature_selection_featurized(METRIC, [test_dataset])

    base_model = xgb.XGBRegressor(objective='reg:squarederror')
    model = RegressorChain(base_model)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    metrics = get_metrics(y_test, preds)
    all_metrics.append(metrics)

accuracies = [m['accuracy'] for m in all_metrics]
print("Average leave-one-out accuracy is: ", np.mean(accuracies))

perc_errors = [m['perc_diff_from_best'] for m in all_metrics]
print("Average leave-one-out percentage_diff_from_best is: ", np.mean(perc_errors))
