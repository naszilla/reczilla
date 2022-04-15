import numpy as np

from Metafeatures.Featurizer import featurize_all_datasets

import catboost as cb
from catboost import Pool, metrics, cv
from sklearn.metrics import accuracy_score

# get features from featurizer as a DataFrame
# TODOs: 1. Ensure feature types are correct
#        2. Missing target variable? We might want to return the name of the target_col from Featurizer
#        3. Missing train-test split?

features, target_col = featurize_all_datasets()

feature_types = features.dtypes

# assert features to be only ints, floats, or strings(categoricals)
assert all([f in (float, int, str) for f in feature_types.unique()])

# assert target_col is of categorical type
assert features[target_col].dtype == str

numerical_features = [col for col, dtype in zip(features, feature_types) if dtype in (int, float)]
categorical_features = [col for col, dtype in zip(features, feature_types) if dtype == str]
categorical_features = [f for f in categorical_features if f != target_col]  # remove target_col from numerical features
categorical_features_indices = [idx for idx, dtype in enumerate(feature_types) if dtype == str]

# form X_train, y_train
all_features = numerical_features + categorical_features
X_train, y_train = features[all_features], features[target_col]

# Model training
model = cb.CatBoostClassifier(
    custom_loss=[metrics.Accuracy()],
    random_seed=42,
    # logging_level='Silent'
)

model.fit(
    X_train, y_train,
    cat_features=categorical_features_indices,
    # eval_set=(X_validation, y_validation),
    # logging_level='Verbose',
    plot=True
);