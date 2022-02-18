import numpy as np
import scipy.sparse as sps
from collections import defaultdict

from Utils.seconds_to_biggest_unit import seconds_to_biggest_unit

from Base.BaseRecommender import BaseRecommender

import surprise
from surprise.trainset import Trainset


class PatchedTrainset(Trainset):
    """
    Subclass of trainset to fix apparent bug in knows_user and knows_item in Surprise.
    """
    def knows_user(self, uid):
        """Indicate if the user is part of the trainset.

        A user is part of the trainset if the user has at least one rating.

        Args:
            uid(int): The (inner) user id. See :ref:`this
                note<raw_inner_note>`.
        Returns:
            ``True`` if user is part of the trainset, else ``False``.
        """

        #return uid in self.ur
        return uid in self.ur and len(self.ur[uid]) > 0

    def knows_item(self, iid):
        """Indicate if the item is part of the trainset.

        An item is part of the trainset if the item was rated at least once.

        Args:
            iid(int): The (inner) item id. See :ref:`this
                note<raw_inner_note>`.
        Returns:
            ``True`` if item is part of the trainset, else ``False``.
        """

        #return iid in self.ir
        return iid in self.ir and len(self.ir[iid]) > 0

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
        if not user_id in ur.keys():
            ur[user_id] = []
        if not item_id in ir.keys():
            ir[item_id] = []
        ur[user_id].append((item_id, rating))
        ir[item_id].append((user_id, rating))

    #return Trainset(ur, ir, n_users, n_items, n_ratings, rating_scale, raw2inner_id_users, raw2inner_id_items)
    return PatchedTrainset(ur, ir, n_users, n_items, n_ratings, rating_scale, raw2inner_id_users, raw2inner_id_items)


class SurpriseAlgoWrapper(BaseRecommender):
    """
    Generic wrapper for a SurpriseAlgorithms algorithm. Enables use of a SurpriseAlgorithms algorithm using the BaseRecommender class
    template.
    """

    def __init__(self, URM_train, verbose=True):
        super(SurpriseAlgoWrapper, self).__init__(URM_train, verbose=verbose)

        self.trainset = URM_to_surprise_trainset(self.URM_train)

    def fit(self, **alg_kwargs):
        """
        :param alg_kwargs: Keyword args to be used to initialize SurpriseAlgorithms algorithm
        :return: None
        """
        # Initialize surprise algorithm class
        self.surprise_model = self.SURPRISE_CLASS(**alg_kwargs)

        # Fit model
        self.surprise_model.fit(self.trainset)

    def _compute_item_score(self, user_id_array, items_to_compute=None, warn_nonparallel=False):
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

    def save_model(self, folder_path, file_name=None):
        raise NotImplementedError("SurpriseAlgoWrapper: save_model not implemented")


class CoClustering(SurpriseAlgoWrapper):
    """Wrapper around surprise.CoClustering"""
    SURPRISE_CLASS = surprise.CoClustering

    def __init__(self, URM_train, verbose=True):
        super(CoClustering, self).__init__(URM_train, verbose=verbose)

        self.unk_items = np.argwhere(np.array(URM_train.sum(axis=0)).squeeze() == 0).squeeze()
        self.unk_users = np.argwhere(np.array(URM_train.sum(axis=1)).squeeze() == 0).squeeze()

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        """
        Provides a parallelized method for co-clustering. Note that there is an apparent bug in the original
        implementation in its handling of unknown items and scores, so the predictions for those cases will be
        inconsistent.

        :param user_id_array:       array containing the user indices whose recommendations need to be computed
        :param items_to_compute:    array containing the items whose scores are to be computed.
                                        If None, all items are computed, otherwise discarded items will have as score -np.inf
        :return:                    array (len(user_id_array), n_items) with the score.
        """
        item_id_array = np.array(range(self.n_items)) if not items_to_compute else items_to_compute

        uc = self.surprise_model.cltr_u[user_id_array]
        ic = self.surprise_model.cltr_i[item_id_array]

        item_scores = (self.surprise_model.avg_cocltr[np.ix_(uc, ic)] +
                       np.expand_dims(
                           self.surprise_model.user_mean[user_id_array] -
                           self.surprise_model.avg_cltr_u[uc],
                           axis=-1) +
                       np.expand_dims(
                           self.surprise_model.item_mean[item_id_array] -
                           self.surprise_model.avg_cltr_i[ic],
                           axis=0))

        unk_user_mask = np.in1d(user_id_array, self.unk_users)
        unk_item_mask = np.in1d(item_id_array, self.unk_items)

        #item_scores[unk_user_mask, :] = np.expand_dims(self.surprise_model.cltr_i[item_id_array], axis=0)
        #item_scores[:, unk_item_mask] = np.expand_dims(self.surprise_model.cltr_u[user_id_array], axis=1)
        item_scores[unk_user_mask, :] = np.expand_dims(self.surprise_model.item_mean[item_id_array], axis=0)
        item_scores[:, unk_item_mask] = np.expand_dims(self.surprise_model.user_mean[user_id_array], axis=1)

        item_scores[np.ix_(unk_user_mask, unk_item_mask)] = self.surprise_model.trainset.global_mean

        # check_scores = np.zeros((len(user_id_array), len(item_id_array)))
        # for user_idx, user in enumerate(user_id_array):
        #     for item_idx, item in enumerate(item_id_array):
        #         check_scores[user_idx, item_idx] = self.surprise_model.estimate(user, item)
        #
        # diff_mat = np.abs(item_scores-check_scores)
        # print(np.max(diff_mat))
        # print(np.max(diff_mat[:, ~unk_item_mask]))

        return item_scores


class SlopeOne(SurpriseAlgoWrapper):
    """Wrapper around surprise.SlopeOne"""
    SURPRISE_CLASS = surprise.SlopeOne

    def __init__(self, URM_train, verbose=True):
        super(SlopeOne, self).__init__(URM_train, verbose=verbose)

        self.unk_items = np.argwhere(np.array(URM_train.sum(axis=0)).squeeze() == 0).squeeze()
        self.unk_users = np.argwhere(np.array(URM_train.sum(axis=1)).squeeze() == 0).squeeze()

        # From CoClustering implementation
        user_mean = np.zeros(self.trainset.n_users, np.double)
        item_mean = np.zeros(self.trainset.n_items, np.double)
        for u in self.trainset.all_users():
            user_mean[u] = np.mean([r for (_, r) in self.trainset.ur[u]])
        for i in self.trainset.all_items():
            item_mean[i] = np.mean([r for (_, r) in self.trainset.ir[i]])

        self.item_mean = item_mean
        self.user_mean = user_mean


    def _compute_item_score(self, user_id_array, items_to_compute=None):
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
        item_id_array = range(self.n_items) if not items_to_compute else items_to_compute
        unk_user_mask = np.in1d(user_id_array, self.unk_users)
        unk_item_mask = np.in1d(item_id_array, self.unk_items)

        freq_mask = self.surprise_model.freq[item_id_array, :] > 0
        has_rating_mask = self.URM_train[user_id_array, :].astype('bool')
        masked_devs = (freq_mask *
                       np.nan_to_num(self.surprise_model.dev[item_id_array, :])).T
        # masked_devs = (freq_mask *
        #                np.nan_to_num(self.surprise_model.dev[item_id_array, :], posinf=np.inf, neginf=-np.inf)).T
        sums = has_rating_mask @ masked_devs
        counts = (has_rating_mask @ freq_mask.T.astype('double'))
        # item_scores = (np.nan_to_num(sums / counts, posinf=np.inf, neginf=-np.inf)
        #                + np.expand_dims(self.user_mean[user_id_array], axis=-1))
        item_scores = (np.nan_to_num(sums / counts)
                       + np.expand_dims(self.user_mean[user_id_array], axis=-1))

        # check_scores = np.zeros(item_scores.shape)
        # for user_idx, user in enumerate(user_id_array):
        #     for item_idx, item in enumerate(item_id_array):
        #         if not (unk_user_mask[user_idx] or unk_item_mask[item_idx]):
        #             check_scores[user_idx, item_idx] = self.surprise_model.estimate(user, item)

        # Fill in missing users and items
        item_scores[unk_user_mask, :] = np.expand_dims(self.item_mean[item_id_array], axis=0)
        item_scores[:, unk_item_mask] = np.expand_dims(self.user_mean[user_id_array], axis=1)
        item_scores[np.ix_(unk_user_mask, unk_item_mask)] = self.surprise_model.trainset.global_mean

        # print(np.max((item_scores - check_scores)[:, ~unk_item_mask]))
        # print(np.sum(np.isnan(item_scores)))

        return item_scores

# Can't deal with missing items/users (apparent limitation in original implementation)
# class SurpriseNMF(SurpriseAlgoWrapper):
#     SURPRISE_CLASS = surprise.NMF
