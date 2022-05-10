import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning)
import pickle
from tqdm import tqdm
from ReczillaClassifier.dataset_families import dataset_family_lookup

RESULTS_DIR = "metadatasets"

def get_metafeats(metadataset_filename):
    """Merge metafeatures and metadataset. Return as dataframe."""
    metadataset_fn = f"{RESULTS_DIR}/{metadataset_filename}.pkl"
    with open(metadataset_fn, "rb") as f:
        meta_dataset = pickle.load(f)

    # single_sample_algs = [
    #     "TopPop",
    #     "GlobalEffects",
    #     "Random",
    #     "SlopeOne",
    # ]
    # min_samples = 30
    #
    # # TODO: we no longer need to select algs based on number of samples.
    # keep_rows = (meta_dataset["num_samples"] >= min_samples) | meta_dataset["alg_name"].isin(single_sample_algs)
    # meta_dataset = meta_dataset.loc[keep_rows, :]

    metafeats_fn = "Metafeatures/Metafeatures.csv"
    metafeats = pd.read_csv(metafeats_fn)
    bucket_prefix = r"gs://reczilla-results/dataset-splits/"
    metafeats["original_split_path"] = bucket_prefix + metafeats["bucket_path"]
    metafeats["dataset_family"] = metafeats["dataset_name"].apply(dataset_family_lookup)
    metafeats.drop(["bucket_path", "split_name", "dataset_name"], axis=1, inplace=True)
    # Joining on full bucket path
    metafeats = meta_dataset.merge(metafeats, on="original_split_path", how='left')

    return metafeats


# Algorithm selection
def rank_algorithms(meta_dataset, test_datasets, metric_name):
    """Compute algorithm ranks for each dataset"""
    # Sanity check to prevent leakage
    for test_dataset in test_datasets:
        assert test_dataset in meta_dataset['dataset_family'].values
    filtered_dataset = meta_dataset[~meta_dataset['dataset_family'].isin(test_datasets)]

    # TODO: instead of ranking algorithms we need to maximize coverage - see sec. 4.1 and equation (1) in overleaf
    all_ranks = []
    for dataset_name, dataset_performance in filtered_dataset.groupby("dataset_name"):
        dataset_performance["rank"] = dataset_performance["max_test_metric_" + metric_name].rank(method='min',
                                                                                                 ascending=False)
        dataset_performance.set_index("alg_name", inplace=True)
        dataset_performance = dataset_performance[["rank"]]
        dataset_performance = dataset_performance.rename(columns={"rank": dataset_name})
        all_ranks.append(dataset_performance)

    ranked_algs = pd.concat(all_ranks, axis=1)
    return ranked_algs


# TODO: instead of finding the best *algorithm*, we need to find the best *alg+hyperparameter* combination (see sec.
#  4.1 in the overleaf). The set of hyperparameters is uniquely identified by column "hyperparameters_source", so we
#  probably should create a new column for this, e.g. "alg_hyperparam", that indicates both the algorithm and the
#  hyperparameter set.
# TODO: instead of ranking algorithms we need to maximize coverage - see sec. 4.1 and equation (1) in overleaf
def select_algs(metafeats, test_datasets, metric_name, num_algs=10):
    """Select num_algs algorithm with best mean rank"""
    ranked_algs = rank_algorithms(metafeats, test_datasets, metric_name)
    return list(ranked_algs.T.mean().sort_values().iloc[:num_algs].index)


# Metafeature selection
def compute_feature_corrs(metafeats, test_datasets, metric_name, selected_algs=None, by_alg_family=False):
    """Compute correlation between each metafeature and the desired metric for all selected algorithms.
    Dataframe result is num_features x num_algorithms."""
    print("Computing correlations...")
    if selected_algs is None:
        if not by_alg_family:
            selected_algs = metafeats["alg_param_name"].unique()
        else:
            selected_algs = metafeats["alg_family"].unique()

    all_features = [col for col in metafeats.columns if col.startswith("f_")]
    # Sanity check to prevent leakage
    for test_dataset in test_datasets:
        assert test_dataset in metafeats['dataset_family'].values
    filtered_metafeats = metafeats[~metafeats['dataset_family'].isin(test_datasets)]

    all_cors = []

    for alg in tqdm(selected_algs):
        if by_alg_family:
            # TODO: Implement algorithm family correlation (if we plan on using it)
            raise NotImplementedError("Algorithm family correlation not yet implemented")
            # filtered_results = filtered_metafeats.loc[(filtered_metafeats["alg_name"] == alg)]
            # alg_cors = filtered_results[all_features].corrwith(filtered_results["max_test_metric_" + metric_name],
            #                                                method="spearman")
        else:
            filtered_results = filtered_metafeats.loc[(filtered_metafeats["alg_param_name"] == alg)]
            alg_cors = filtered_results[all_features].corrwith(filtered_results["test_metric_" + metric_name],
                                                               method="spearman")

        alg_cors.name = alg
        all_cors.append(alg_cors)
    all_cors = pd.concat(all_cors, axis=1).abs()
    return all_cors


def select_features(metafeats, test_datasets, metric_name, selected_algs=None, num_feats=10):
    """Select num_feats features. Greedy scheme. At each step, we compute the best correlations
    across all metafeatures for each algorithm so far. We add whichever metafeature can obtain the maximum
    improvement across any single one of the best correlations for the selected algorithms."""
    all_cors = compute_feature_corrs(metafeats, test_datasets, metric_name, selected_algs=selected_algs)
    selected_feats = [all_cors.max(axis=1).idxmax()]

    while len(selected_feats) < num_feats:
        # Current best correlations
        current_best_cors = all_cors.loc[selected_feats].max(axis=0)
        # Pick whichever feature results in the highest maximum improvement on the current best correlations
        selected_feats.append((
                                      all_cors.loc[~all_cors.index.isin(selected_feats)] - current_best_cors)
                              .max(axis=1)
                              .idxmax())
    return selected_feats


# TODO: pass number of algs & number of meta-features as an arg to this function
def alg_feature_selection_featurized(metric_name, test_datasets, dataset_name, train_datasets=None):
    # TODO: Filter based on minimum number of alg_param_name samples?
    metafeats = get_metafeats(dataset_name)

    if train_datasets is not None:
        # TODO: The functionality of this line might be broken
        metafeats = metafeats[metafeats['dataset_family'].isin(train_datasets + test_datasets)]

    # TODO: This function to be updated
    selected_algs = select_algs(metafeats, test_datasets, metric_name)
    selected_feats = select_features(metafeats, test_datasets, metric_name, selected_algs)
    
    ##### Featurization
    final_feat_columns = selected_feats
    X_train = metafeats[metafeats['alg_name'].isin(selected_algs) & ~metafeats['dataset_family'].isin(test_datasets)]

    metric_col_name = "max_test_metric_" + metric_name
    X_train = X_train[[metric_col_name] + ["dataset_name", "alg_name"] + final_feat_columns]

    transforms = {f: 'last' for f in final_feat_columns}
    transforms.update({metric_col_name: list, 'alg_name': list})

    X_train_grouped = X_train.groupby('dataset_name').agg(transforms)

    def get_ordered_target(row):
        avg = np.mean(row[metric_col_name])
        algos_perfs = {alg: val for val, alg in zip(row[metric_col_name], row['alg_name'])}
        algos_perfs.update({alg: avg for alg in selected_algs if alg not in algos_perfs})
        ordered_target = [algos_perfs[key] for key in sorted(algos_perfs.keys(), reverse=True)]
        return ordered_target

    X_train_grouped['target'] = X_train_grouped.apply(get_ordered_target, axis=1)

    X_train = X_train_grouped[final_feat_columns].values
    y_train = np.array(X_train_grouped['target'].to_list())
    
    test_data = metafeats[metafeats['dataset_family'].isin(test_datasets) & metafeats['alg_name'].isin(selected_algs)]
    test_data = test_data[[metric_col_name] + ["dataset_name", "alg_name"] + final_feat_columns]

    X_test = test_data[final_feat_columns].iloc[0].values
    X_test = np.array([X_test])
    y_test = test_data.groupby('dataset_name').agg(transforms).apply(get_ordered_target, axis=1).values
    # y_test = y_test.tolist()
    
    return X_train, y_train, X_test, y_test

# Test code
if __name__ == '__main__':
    metafeats = get_metafeats("metadata-v0")