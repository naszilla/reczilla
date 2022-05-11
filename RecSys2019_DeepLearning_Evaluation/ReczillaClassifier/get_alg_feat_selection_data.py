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


def select_algs(metafeats, dataset_family_list, metric_name, num_algs=10):
    """
    Select a set of parameterized algorithms with good coverage over all datasets belonging to one or mode dataset
    families. Algorithms are selected "greedily" to maximize coverage (see eq. 1 in the paper).

    For coding purposes, higher metrics = better. This may be different from the paper.

    args:
    - metafeats (dataframe): contains performance metrics and dataset meta-features
    - dataset_family_list (list[str]): in all calculations, only include datasets belonging to the families listed here
    - metric_name (str): name of the metric used to select algs. this must be a column in the meta-dataset
    - num_algs (int): number of algorithms to select

    output:
    - a list of parameterized algorithms, from the column alg_param_name in the meta-dataset
    """

    if metric_name not in metafeats.columns:
        raise Exception(f"metric_name {metric_name} not found in metafeats dataframe.")

    # only include dataset families
    # create a temporary df for use in this function
    tmp_df = metafeats.loc[metafeats["dataset_family"].isin(dataset_family_list), :].copy()

    # require that there is only one result for each dataset + parameterized alg pair. if not, drop duplicates (keep first).
    pair_counts = tmp_df.groupby(["original_split_path", "alg_param_name"]).size().rename("size").reset_index()

    if pair_counts["size"].max() > 1:
        e = f"multiple rows found for dataset + parameterized pairs:\n{str(pair_counts[pair_counts['size'] > 1])}"
        raise Exception(e)

    if len(tmp_df) == 0:
        raise Exception(f"no rows found for dataset_family_list = {dataset_family_list}")

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

    # function for calculating coverage
    def coverage(alg_subset):
        # return the average pct_diff_from_opt for the *best* parameterized alg in the subset, over all datasets
        # this assumes that there is exactly one metric for each dataset + parameterized alg (we check for this above)
        subset_rows = tmp_df["alg_param_name"].isin(alg_subset)

        best_pct_diff = []
        for dataset in all_datasets:
            # find all rows for this dataset and this subset
            dataset_rows = tmp_df["original_split_path"] == dataset
            eval_rows = tmp_df.loc[dataset_rows & subset_rows, :]
            if len(eval_rows) == 0:
                best_pct_diff.append(np.nan)  # no result for this dataset...
            else:
                best_pct_diff.append(eval_rows["pct_diff_opt"].max())  # add the best (max) pct_diff_opt

        return best_pct_diff


    ###################################
    # greedy algorithm subset selection

    selected_algs = []
    candidate_algs = all_algs.copy()  # all algs that we can select from

    for i_step in range(num_algs):
        # print(f"[select_algs] beginning step {i_step + 1} of {num_algs}")
        if len(candidate_algs) == 0:
            raise Exception("no candidate algs left to select from.")

        # add the algorithm that results in the largest coverage for the subset
        avg_coverage_list = []  # avg. coverage over all datasets
        sum_coverage_list = []  # sum of coverage over all datasets (for tiebreaking)
        for i_alg, alg_name in enumerate(candidate_algs):
            # calculate the average coverage over all datasets
            cov = coverage(selected_algs + [alg_name])
            avg_coverage_list.append(np.mean(cov))  # mean of covg over all datasets, and nan if any are nan
            sum_coverage_list.append(np.nansum(cov))  # sum of covg over all datasets, ignoring nan

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
        del candidate_algs[add_index]

    return selected_algs


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
    # get metafeatures
    metafeats = get_metafeats("metadata-v0")

    # test select algs
    metric_name = 'test_metric_ARHR_ALL_HITS_cut_10'
    dataset_family_list = list(metafeats["dataset_family"].unique())
    selected_algs = select_algs(metafeats, dataset_family_list, metric_name, num_algs=4)
    print(f"selected_algs: {selected_algs}")
    # example output:
    # selected_algs: ['ItemKNNCF:dice_random_0', 'ItemKNNCF:dice_random_41', 'TopPop:default',
    #                 'ItemKNNCF:euclidean_random_19']
