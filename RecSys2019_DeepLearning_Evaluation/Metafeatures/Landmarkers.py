from Metafeatures.utils import register_func

from Data_manager.split_functions.split_train_validation_leave_k_out import split_train_leave_k_out_user_wise
from Base.Evaluation.Evaluator import EvaluatorHoldout

# Algorithms to landmark
from Base.NonPersonalizedRecommender import TopPop
from KNN.UserKNNCFRecommender import UserKNNCFRecommender
from KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from MatrixFactorization.PureSVDRecommender import PureSVDRecommender

import numpy as np
import scipy

feature_func_lookup = {}

# Entries are alg_name: (alg_class, fit_kwargs)
landmark_algs = {
    "top_pop":
        (TopPop,
         {}),
    "item_knn_k1":
        (ItemKNNCFRecommender,
         {"topK": 1}),
    "user_knn_k1":
        (UserKNNCFRecommender,
         {"topK": 1}),
    "pure_svd_fact1":
        (PureSVDRecommender,
         {"num_factors": 1}),
    "item_knn_k5":
        (ItemKNNCFRecommender,
         {"topK": 5}),
    "user_knn_k5":
        (UserKNNCFRecommender,
         {"topK": 5}),
    "pure_svd_fact5":
        (PureSVDRecommender,
         {"num_factors": 5})
}

feature_list = [
    ("landmarker", {"alg": alg_name}) for alg_name in landmark_algs.keys()
]

# Select =items that are not cold. If less than min_items warm items exist,
# picks a random subset of the cold items to achieve the min.
def select_warm_items(matrix, min_items):
    mask = matrix.getnnz(axis=0) > 0
    item_idcs = np.argwhere(mask).squeeze()
    # Pick some of the cold items if needed
    if len(item_idcs) < min_items:
        cold_idcs = np.argwhere(~mask).squeeze()
        item_idcs = np.concatenate((item_idcs, np.random.choice(cold_idcs, size=min_items-len(item_idcs), replace=False)))
    return matrix[:, item_idcs]

# Randomly select items from matrix. Ensures each user
# has at least two items, if possible
def random_item_selection(matrix, num_items, min_items_per_user=2):
    if not type(matrix) is scipy.sparse.csr.csr_matrix:
        matrix = matrix.tocsr()
    nonzero_item_idcs = np.split(matrix.indices, matrix.indptr[1:-1])
    min_viable_idcs = set()
    for user_item_indices in nonzero_item_idcs:
        new_indeces = np.random.choice(user_item_indices, size=min_items_per_user, replace=False)
        min_viable_idcs.update(new_indeces)

    # Add random items if there aren't enough already
    if len(min_viable_idcs) < num_items:
        candidates = list(set(range(matrix.shape[1])) - min_viable_idcs)
        new_elems = np.random.choice(candidates, size=num_items-len(min_viable_idcs), replace=False)
        item_idcs = list(min_viable_idcs) + list(new_elems)
    else:
        item_idcs = list(min_viable_idcs)

    return matrix[:, item_idcs]

# Landmark a basic algorithm
@register_func(feature_func_lookup)
def landmarker(train_set, alg, random_seed=42):
    np.random.seed(random_seed)

    n_users_subsample = 100 # Number of users in the subsample
    n_items_subsample = 250 # Number of items in the subsample (may be slightly more to ensure no cold users in train set)
    min_ratings_train = 1 # Minimum number of ratings that we need on the training set
    max_subsample_attempts = 20 # Max number of times to attempt subsampling
    cutoff_list = [1, 5]
    min_items = max(cutoff_list) + 1  # Minimum number of items that we need to keep for evaluation

    if train_set.shape[1] < min_items:
        raise RuntimeError("Cannot use a cutoff of {} for dataset with {} items".format(
            min_items-1, train_set.shape[1]))

    # To be able to split, we need to ensure there is a number of users with more than one item.
    # We also need the items for those users.
    mult_item_users = np.argwhere(train_set.getnnz(axis=1) > 1).squeeze()
    masked_data = train_set[mult_item_users, :]
    masked_data = select_warm_items(masked_data, min_items)
    n_users, n_items = masked_data.shape

    for attempt in range(max_subsample_attempts):
        user_idcs = np.random.choice(n_users, size=min(n_users, n_users_subsample), replace=False)
        subsample = masked_data[user_idcs, :]
        subsample = select_warm_items(subsample, min_items)
        current_n_items = subsample.shape[1]

        if current_n_items > n_items_subsample:
            subsample = random_item_selection(subsample, n_items_subsample, min_items_per_user=2)

        sub_train, sub_val = split_train_leave_k_out_user_wise(subsample,
                                                               k_out=1,
                                                               use_validation_set=False,
                                                               leave_random_out=True)
        if sub_train.nnz >= min_ratings_train:
            break
        elif attempt == max_subsample_attempts-1:
            print("Error: did not achieve number of subset training sample within {} attempts.".format(
                max_subsample_attempts))
            import pdb
            pdb.set_trace()


    evaluator_validation = EvaluatorHoldout(
        sub_val, cutoff_list, exclude_seen=False
    )

    alg_class, alg_kwargs = landmark_algs[alg]
    recommender = alg_class(sub_train)
    recommender.fit(**alg_kwargs)

    results_dict, _ = evaluator_validation.evaluateRecommender(recommender)

    consolidated_results = {}
    for cutoff_val, metrics in results_dict.items():
        consolidated_results.update({"cut_{}__{}".format(cutoff_val, metric_name): metric_val
                                     for metric_name, metric_val in metrics.items()})

    return consolidated_results
