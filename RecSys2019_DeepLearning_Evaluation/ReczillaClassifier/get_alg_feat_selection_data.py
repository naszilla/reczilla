import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning)
import pickle
from tqdm import tqdm
from ReczillaClassifier.dataset_families import dataset_family_lookup
from ReczillaClassifier.fixed_algs_feats import SELECTED_FEATS_100, SELECTED_ALGS_100
from functools import lru_cache
from datetime import datetime
import random

RESULTS_DIR = "metadatasets"

@lru_cache(maxsize=None)
def get_metafeats(metadataset_filename):
    """Merge metafeatures and metadataset. Return as dataframe."""
    metadataset_fn = f"{RESULTS_DIR}/{metadataset_filename}.pkl"
    with open(metadataset_fn, "rb") as f:
        meta_dataset = pickle.load(f)  # TODO: this file is intended to be read using pd.read_pickle(metadataset_fn). pickle.load may or may not work as expected

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

def my_print(x, verbose):
    if verbose:
        print(x)

def select_algs(metafeats, exclude_dataset_families, metric_name, num_algs=10, verbose=False):
    """
    Select a set of parameterized algorithms with good coverage over all datasets, excluding some dataset families.
    Algorithms are selected "greedily" to maximize coverage (see eq. 1 in the paper).

    For coding purposes, higher metrics = better. This may be different from the paper.

    args:
    - metafeats (dataframe): contains performance metrics and dataset meta-features
    - exclude_dataset_families (list[str]): a list of dataset families to exclude
    - metric_name (str): name of the metric used to select algs. this must be a column in the meta-dataset
    - num_algs (int): number of algorithms to select

    output:
    - a list of parameterized algorithms, from the column alg_param_name in the meta-dataset
    """

    if metric_name not in metafeats.columns:
        raise Exception(f"metric_name {metric_name} not found in metafeats dataframe.")

    # Sanity check to prevent leakage
    if exclude_dataset_families is not None:
        for family_name in exclude_dataset_families:
            assert family_name in metafeats['dataset_family'].values

    # exclude some dataset families
    # create a temporary df for use in this function
    if exclude_dataset_families is None:
        tmp_df = metafeats.copy()
    else:
        tmp_df = metafeats.loc[~metafeats['dataset_family'].isin(exclude_dataset_families), :].copy()

    # require that there is only one result for each dataset + parameterized alg pair. if not, drop duplicates (keep first).
    pair_counts = tmp_df.groupby(["original_split_path", "alg_param_name"]).size().rename("size").reset_index()

    if pair_counts["size"].max() > 1:
        e = f"multiple rows found for dataset + parameterized pairs:\n{str(pair_counts[pair_counts['size'] > 1])}"
        raise Exception(e)

    if len(tmp_df) == 0:
        raise Exception(f"no rows found after excluding dataset_family_list {dataset_family_list}")

    # for all datasets, find the best (maximum) metric achieved by any parameterized algs
    # all datasets are uniquely identified by their original path on gcloud (original_split_path)
    dataset_best_metrics = tmp_df.groupby("original_split_path")[metric_name].max().rename("max_metric").reset_index()

    # get all datasets & algs
    all_datasets = list(tmp_df["original_split_path"].unique())
    all_algs = list(tmp_df["alg_param_name"].unique())

    # merge in the best metrics for each dataset
    tmp_df = tmp_df.merge(dataset_best_metrics, on="original_split_path", how="left")

    # calculate the pct-difference-from-best for each parameterized alg (row) in the meta-dataset
    tmp_df.loc[:, "pct_diff_opt"] = 100.0 * (tmp_df[metric_name].values - tmp_df["max_metric"].values) / tmp_df["max_metric"].values

    # create a list of performance on each dataset (ordered by list "all_datasets") , for all parameterized algs
    performance_dict = dict()
    for alg_param in tmp_df["alg_param_name"].unique():
        alg_df = tmp_df.loc[tmp_df["alg_param_name"] == alg_param, ["original_split_path", "pct_diff_opt"]].set_index("original_split_path").to_dict()
        performance_dict[alg_param] = np.array([alg_df["pct_diff_opt"].get(dataset, np.nan) for dataset in all_datasets])

    def coverage_fast(old_coverage, alg_param):
        """compare the old coverage to the alg coverage, using performance_dict"""
        return np.nanmax(np.stack([old_coverage, performance_dict[alg_param]]), axis=0)

    ###################################
    # greedy algorithm subset selection

    selected_algs = []
    candidate_algs = all_algs.copy()  # all algs that we can select from

    # keep track of the current coverage
    current_coverage = np.array([np.nan for _ in all_datasets])

    for i_step in range(num_algs):
        my_print(f"[select_algs] beginning step {i_step + 1} of {num_algs}", verbose)
        if len(candidate_algs) == 0:
            raise Exception("no candidate algs left to select from.")

        # add the algorithm that results in the largest coverage for the subset
        cov_list = []
        avg_coverage_list = []  # avg. coverage over all datasets
        sum_coverage_list = []  # sum of coverage over all datasets (for tiebreaking)
        for i_alg, alg_name in enumerate(candidate_algs):
            # calculate the average coverage over all datasets
            tmp_cov = coverage_fast(current_coverage, alg_name)
            avg_coverage_list.append(np.mean(tmp_cov))  # mean of covg over all datasets, and nan if any are nan
            sum_coverage_list.append(np.nansum(tmp_cov))  # sum of covg over all datasets, ignoring nan
            cov_list.append(tmp_cov)

        # if some algs have non-nan avg-coverage, select the alg with the greatest coverage
        if any(np.array(avg_coverage_list) != np.nan):
            # add the alg that results in the largest coverage, ignoring nans
            add_index = np.nanargmax(avg_coverage_list)
        elif any(np.array(sum_coverage_list) != np.nan):
            # ... otherwise, add the alg with the largest sum of coverage, ignoring nans
            add_index = np.nanargmax(sum_coverage_list)
        else:
            print(f"[select_algs] WARNING: no coverage during step {i_step+1}. adding a random algorithm")
            add_index = np.random.randint(0, len(candidate_algs))

        selected_algs.append(candidate_algs[add_index])
        current_coverage = cov_list[add_index]
        del candidate_algs[add_index]

    my_print(f"done. selected algs: {selected_algs}", verbose)
    return selected_algs

# define functions for weighted correlation
# from https://stackoverflow.com/questions/38641691/weighted-correlation-coefficient-with-pandas
def w_cov(x, y, w):
    """Weighted Covariance"""
    return np.sum(w * (x - np.average(x, weights=w)) * (y - np.average(y, weights=w))) / np.sum(w)

def weighted_corr(x, y, w):
    """Weighted Correlation"""
    return w_cov(x, y, w) / np.sqrt(w_cov(x, x, w) * w_cov(y, y, w))

# Metafeature selection
def compute_feature_corrs(metafeats, exclude_dataset_families, metric_name, selected_algs=None, by_alg_family=False,
                          filter_nans=True):
    """Compute correlation between each metafeature and the desired metric for all selected algorithms.
    Dataframe result is num_features x num_algorithms."""
    print("Computing correlations...")
    if selected_algs is None:
        if not by_alg_family:
            selected_algs = metafeats["alg_param_name"].unique()
        else:
            selected_algs = metafeats["alg_family"].unique()

    all_features = [col for col in metafeats.columns if col.startswith("f_")]
    if filter_nans:
        nan_counts = metafeats[[col for col in metafeats.columns if col.startswith("f_")]].isna().sum(axis=0)
        exclude_feats = nan_counts[nan_counts > 0].index
        all_features = [feat for feat in all_features if feat not in exclude_feats]

    # Sanity check to prevent leakage
    if exclude_dataset_families is not None:
        for family_name in exclude_dataset_families:
            assert family_name in metafeats['dataset_family'].values
        filtered_metafeats = metafeats[~metafeats['dataset_family'].isin(exclude_dataset_families)]
    else:
        filtered_metafeats = metafeats

    all_cors = []

    for alg in selected_algs:
        if by_alg_family:
            # TODO: Implement algorithm family correlation (if we plan on using it)
            raise NotImplementedError("Algorithm family correlation not yet implemented")
            # filtered_results = filtered_metafeats.loc[(filtered_metafeats["alg_name"] == alg)]
            # alg_cors = filtered_results[all_features].corrwith(filtered_results["max_test_metric_" + metric_name],
            #                                                method="spearman")
        else:
            filtered_results = filtered_metafeats.loc[(filtered_metafeats["alg_param_name"] == alg)]
            frequencies = filtered_results['dataset_family'].map(filtered_results['dataset_family'].value_counts())
            freq_weighted_corr = lambda x, y: weighted_corr(x, y, 1.0 / frequencies.values)
            alg_cors = filtered_results[all_features].corrwith(filtered_results[metric_name], method=freq_weighted_corr)

        alg_cors.name = alg
        all_cors.append(alg_cors)
    all_cors = pd.concat(all_cors, axis=1).abs()
    return all_cors


def select_features(metafeats, test_datasets, metric_name, selected_algs=None, num_feats=10, filter_nans=True):
    """Select num_feats features. Greedy scheme. At each step, we compute the best correlations
    across all metafeatures for each algorithm so far. We add whichever metafeature can obtain the maximum
    improvement across any single one of the best correlations for the selected algorithms."""
    all_cors = compute_feature_corrs(metafeats, test_datasets, metric_name, selected_algs=selected_algs)
    if filter_nans:
        nan_counts = metafeats[[col for col in metafeats.columns if col.startswith("f_")]].isna().sum(axis=0)
        exclude_feats = nan_counts[nan_counts > 0].index
        all_cors = all_cors.loc[~all_cors.index.isin(exclude_feats)]

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

def filter_for_cunha(selected_algs, selected_feats, compare_cunha):
    
    assert compare_cunha in ('cf4cf', 'cunha-2018')

    if compare_cunha == 'cf4cf':
        algs = [alg for alg in selected_algs if 'default' in alg]
        feats = [feat for feat in selected_feats]
    elif compare_cunha == 'cunha-2018':
        algs = [alg for alg in selected_algs if 'default' in alg]
        feats = [feat for feat in selected_feats if '_landmarker' not in feat]
    return algs, feats


def alg_feature_selection_featurized(metric_name, test_datasets, dataset_name, train_datasets=None,
                                     fixed_algs_feats=False, num_algs=10, num_feats=10, random_algs=False,
                                     random_feats=False, compare_cunha=None, get_extra_outputs=False):
    
    # TODO: num_algs and num_feats parameters are currently only implemented for fixed_alg_feats=True
    if not fixed_algs_feats and (num_algs != 10 or num_feats != 10):
        print('WARNING: alg/feat selection are hard-coded to 10!')
    
    # TODO: Filter based on minimum number of alg_param_name samples?
    metafeats = get_metafeats(dataset_name)

    exclude_test_dataset_families = [dataset_family_lookup(test_dataset) for test_dataset in test_datasets]

    if train_datasets is not None:
        # TODO: The functionality of this line might be broken
        train_dataset_families = [dataset_family_lookup(train_dataset) for train_dataset in train_datasets]
        metafeats = metafeats[metafeats['dataset_family'].isin(train_dataset_families + exclude_test_dataset_families)]
    
    fixed_algs = SELECTED_ALGS_100.copy()
    fixed_feats = SELECTED_FEATS_100.copy()

    if compare_cunha:
        if fixed_algs_feats:
            fixed_algs, fixed_feats = filter_for_cunha(fixed_algs, fixed_feats, compare_cunha)
        else:
            raise NotImplementedError

    time = datetime.now()
    # TODO: This function to be updated
    print("selecting algs and features..")
    selected_algs = select_algs(metafeats, exclude_test_dataset_families, metric_name) if not fixed_algs_feats else fixed_algs[:num_algs]
    if random_algs:
        selected_algs = fixed_algs[:40]
        random.shuffle(selected_algs)
        selected_algs = selected_algs[:num_algs]
    print("done selecting algs in : ", datetime.now() - time)

    time = datetime.now()
    selected_feats = select_features(metafeats, exclude_test_dataset_families, metric_name, selected_algs) if not fixed_algs_feats else fixed_feats[:num_feats]
    if random_feats:
        selected_feats = fixed_feats[:40]
        random.shuffle(selected_feats)
        selected_feats = selected_feats[:num_feats]

    print("done selecting features in : ", datetime.now() - time)
    # print(test_datasets, num_algs, selected_algs, selected_feats)
    
    ##### Featurization
    # TODO: Group by original_split_path instead
    best_performing = metafeats.groupby("dataset_name")[metric_name].max()
    worst_performing = metafeats.groupby("dataset_name")[metric_name].min()

    final_feat_columns = selected_feats
    X_train = metafeats[metafeats['alg_param_name'].isin(selected_algs) & ~metafeats['dataset_family'].isin(exclude_test_dataset_families)]

    # TODO: This line to be updated (should just be metric_name), like this:
    # metric_col_name = metric_name. DONE: (Sujay)
    metric_col_name = metric_name
    X_train = X_train[[metric_col_name] + ["dataset_name", "alg_param_name"] + final_feat_columns]

    transforms = {f: 'last' for f in final_feat_columns}
    transforms.update({metric_col_name: list, 'alg_param_name': list})

    # TODO: group by original_split_path instead
    X_train_grouped = X_train.groupby('dataset_name').agg(transforms)

    def get_ordered_target(row):
        avg = np.mean(row[metric_col_name])
        algos_perfs = {alg: val for val, alg in zip(row[metric_col_name], row['alg_param_name'])}
        algos_perfs.update({alg: avg for alg in selected_algs if alg not in algos_perfs})
        ordered_target = [algos_perfs[key] for key in sorted(algos_perfs.keys(), reverse=True)]
        return ordered_target

    X_train_grouped['target'] = X_train_grouped.apply(get_ordered_target, axis=1)

    X_train = X_train_grouped[final_feat_columns].values
    y_train = np.array(X_train_grouped['target'].to_list())

    if test_datasets:
        test_data = metafeats[metafeats['dataset_name'].isin(test_datasets) & metafeats['alg_param_name'].isin(selected_algs)]
        test_data = test_data[[metric_col_name] + ["dataset_name", "alg_param_name"] + final_feat_columns]

        X_test_grouped = test_data.groupby('dataset_name').agg(transforms)
        X_test_grouped['target'] = X_test_grouped.apply(get_ordered_target, axis=1)

        X_test = X_test_grouped[final_feat_columns].values
        y_test = np.array(X_test_grouped['target'].to_list())
        # y_test = y_test.tolist()
        y_best_test = best_performing.loc[X_test_grouped.index].values
        y_worst_test = worst_performing.loc[X_test_grouped.index].values
        y_range_test = np.stack([y_best_test, y_worst_test], axis=-1)
    else:
        X_test, y_test, y_range_test = None, None, None

    if not get_extra_outputs:
        return X_train, y_train, X_test, y_test, y_range_test
    else:
        extra_outputs = {"selected_algs": selected_algs,
                         "selected_feats": selected_feats}
        return X_train, y_train, X_test, y_test, y_range_test, extra_outputs

# Test code
if __name__ == '__main__':
    # get metafeatures
    metafeats = get_metafeats("metadata-v0")

    # test select algs
    metric_name = 'test_metric_PRECISION_cut_10'
    # dataset_family_list = list(metafeats["dataset_family"].unique())
    selected_algs = select_algs(metafeats, None, metric_name, num_algs=100, verbose=True)
    # print(f"selected_algs: {selected_algs}")
    ## example output:
    # [select_algs] beginning step 1 of 4
    # [select_algs] beginning step 2 of 4
    # [select_algs] beginning step 3 of 4
    # [select_algs] beginning step 4 of 4
    # done. selected algs: ['ItemKNNCF:dice_random_0', 'ItemKNNCF:dice_random_41', 'TopPop:default', 'ItemKNNCF:euclidean_random_19']
    # selected_algs: ['ItemKNNCF:dice_random_0', 'ItemKNNCF:dice_random_41', 'TopPop:default', 'ItemKNNCF:euclidean_random_19']

    feats = select_features(metafeats, None, metric_name, selected_algs=selected_algs)
    print("features:")
    print(feats)