import numpy as np
import scipy.sparse as sps
from collections import defaultdict

from Utils.seconds_to_biggest_unit import seconds_to_biggest_unit

from Base.BaseRecommender import BaseRecommender

import surprise
from surprise.trainset import Trainset

def URM_to_surprise_trainset(URM_train):
    """
    Convert dataset representation to surprise representation.
    :param URM_train: Sparse matrix of shape (users, items)
    :return: SurpriseAlgorithms Trainset object with interactions
    """
    n_users, n_items = URM_train.shape
    n_ratings = URM_train.nnz
    rating_scale = (URM_train.data.min(), URM_train.data.max())
    raw2inner_id_users = {idx: idx for idx in range(n_users)}
    raw2inner_id_items = {idx: idx for idx in range(n_items)}
    rows, cols = URM_train.nonzero()

    ur = defaultdict(list)
    ir = defaultdict(list)
    for user_id, item_id, rating in zip(rows, cols, URM_train.data):
        ur[user_id].append((item_id, rating))
        ir[item_id].append((user_id, rating))

    return Trainset(ur, ir, n_users, n_items, n_ratings, rating_scale, raw2inner_id_users, raw2inner_id_items)


class SurpriseAlgoWrapper(BaseRecommender):
    """
    Generic wrapper for a SurpriseAlgorithms algorithm. Enables use of a SurpriseAlgorithms algorithm using the BaseRecommender class
    template.
    """
    def __init__(self, URM_train, verbose = True):
        super(SurpriseAlgoWrapper, self).__init__(URM_train, verbose = verbose)


    def fit(self, **alg_kwargs):
        """
        :param alg_kwargs: Keyword args to be used to initialize SurpriseAlgorithms algorithm
        :return: None
        """

        trainset = URM_to_surprise_trainset(self.URM_train)

        # Initialize surprise algorithm class
        self.surprise_model = self.SURPRISE_CLASS(**alg_kwargs)

        # Fit model
        self.surprise_model.fit(trainset)

    def _compute_item_score(self, user_id_array, items_to_compute = None, warn_nonparallel=False):
        """
        Compute an array of item scores. Note that SurpriseAlgorithms algorithms do not provide predictions in a parallelized
        manner. This function provides a generic wrapper around the surprise estimate() method into the
        _compute_item_score from BaseRecommender. If faster execution is required, implement a parallelized method for
        the chosen SurpriseAlgorithms algorithm and override this method.

        :param user_id_array:       array containing the user indices whose recommendations need to be computed
        :param items_to_compute:    array containing the items whose scores are to be computed.
                                        If None, all items are computed, otherwise discarded items will have as score -np.inf
        :return:                    array (len(user_id_array), n_items) with the score.
        """
        if warn_nonparallel:
            print("Warning: Using non-parallelized method of base SurpriseAlgoWrapper")
        item_id_array = range(self.n_items) if not items_to_compute else items_to_compute
        item_scores = np.zeros((len(user_id_array), len(item_id_array)))

        for user_idx, user in enumerate(user_id_array):
            for item_idx, item in enumerate(item_id_array):
                item_scores[user_idx, item_idx] = self.surprise_model.estimate(user, item)

        return item_scores

    def save_model(self, folder_path, file_name = None):
        raise NotImplementedError("SurpriseAlgoWrapper: save_model not implemented")

class CoClustering(SurpriseAlgoWrapper):
    """Wrapper around surprise.CoClustering"""
    SURPRISE_CLASS = surprise.CoClustering

class SlopeOne(SurpriseAlgoWrapper):
    """Wrapper around surprise.SlopeOne"""
    SURPRISE_CLASS = surprise.SlopeOne

# Can't deal with missing items/users (apparent limitation in original implementation)
# class SurpriseNMF(SurpriseAlgoWrapper):
#     SURPRISE_CLASS = surprise.NMF