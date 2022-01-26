#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Maurizio Ferrari Dacrema, Massimo Quadrana
"""


import numpy as np
import unittest
import scipy.sparse as sps


class _Metrics_Object(object):
    """
    Abstract class that should be used as superclass of all metrics requiring an object, therefore a state, to be computed
    """
    def __init__(self):
        pass

    def __str__(self):
        return "{:.4f}".format(self.get_metric_value())

    def add_recommendations(self, recommended_items_ids):
        raise NotImplementedError()

    def get_metric_value(self):
        raise NotImplementedError()

    def merge_with_other(self, other_metric_object):
        raise NotImplementedError()


####################################################################################################################
###############                 ACCURACY METRICS
####################################################################################################################


class MAP(_Metrics_Object):
    """
    Mean Average Precision, defined as the mean of the AveragePrecision over all users

    """

    def __init__(self):
        super(MAP, self).__init__()
        self.cumulative_AP = 0.0
        self.n_users = 0

    def add_recommendations(self, is_relevant, pos_items):
        self.cumulative_AP += average_precision(is_relevant)
        self.n_users += 1

    def get_metric_value(self):
        return self.cumulative_AP/self.n_users

    def merge_with_other(self, other_metric_object):
        assert other_metric_object is MAP, "MAP: attempting to merge with a metric object of different type"

        self.cumulative_AP += other_metric_object.cumulative_AP
        self.n_users += other_metric_object.n_users



def average_precision(is_relevant):

    if len(is_relevant) == 0:
        a_p = 0.0
    else:
        p_at_k = is_relevant * np.cumsum(is_relevant, dtype=np.float64) / (1 + np.arange(is_relevant.shape[0]))
        a_p = np.sum(p_at_k) / is_relevant.shape[0]

    assert 0 <= a_p <= 1, a_p
    return a_p




class MAP_MIN_DEN(_Metrics_Object):
    """
    Mean Average Precision, defined as the mean of the AveragePrecision over all users
    The AveragePrecision's denominator is capped at the positive items number for that user

    """

    def __init__(self):
        super(MAP_MIN_DEN, self).__init__()
        self.cumulative_AP = 0.0
        self.n_users = 0

    def add_recommendations(self, is_relevant, pos_items):
        self.cumulative_AP += average_precision_min_denominator(is_relevant, pos_items)
        self.n_users += 1

    def get_metric_value(self):
        return self.cumulative_AP/self.n_users

    def merge_with_other(self, other_metric_object):
        assert other_metric_object is MAP_MIN_DEN, "MAP_MIN_DEN: attempting to merge with a metric object of different type"

        self.cumulative_AP += other_metric_object.cumulative_AP
        self.n_users += other_metric_object.n_users



def average_precision_min_denominator(is_relevant, pos_items):

    if len(is_relevant) == 0:
        a_p = 0.0
    else:
        p_at_k = is_relevant * np.cumsum(is_relevant, dtype=np.float64) / (1 + np.arange(is_relevant.shape[0]))
        a_p = np.sum(p_at_k) / np.min([pos_items.shape[0], is_relevant.shape[0]])

    assert 0 <= a_p <= 1, a_p
    return a_p



class MRR(_Metrics_Object):
    """
    Mean Reciprocal Rank, defined as the mean of the Reciprocal Rank over all users
    The reciprocal rank is 1 divided by the position of the first relevant item

    """

    def __init__(self):
        super(MRR, self).__init__()
        self.cumulative_RR = 0.0
        self.n_users = 0

    def add_recommendations(self, is_relevant):
        self.cumulative_RR += rr(is_relevant)
        self.n_users += 1

    def get_metric_value(self):
        return self.cumulative_RR/self.n_users

    def merge_with_other(self, other_metric_object):
        assert other_metric_object is MRR, "MRR: attempting to merge with a metric object of different type"

        self.cumulative_RR += other_metric_object.cumulative_RR
        self.n_users += other_metric_object.n_users



def rr(is_relevant):
    """
    Reciprocal rank of the FIRST relevant item in the ranked list (0 if none)
    :param is_relevant: boolean array
    :return:
    """

    ranks = np.arange(1, len(is_relevant) + 1)[is_relevant]

    if len(ranks) > 0:
        return 1. / ranks[0]
    else:
        return 0.0





class HIT_RATE(_Metrics_Object):
    """
    Hit Rate, defined as the quota of users that received at least a correct recommendation.
    It has values between 0 and 1. It strictly increases as the recommendation list length increases.

    Note that this is different w.r.t. COVERAGE_USER_HIT, COVERAGE_USER_HIT uses as denominator
    all the users in the dataset, HR only those for which a recommendation was computed.
    In this framework if a user has no possible correct recommendations then it is skipped.
    Therefore HR = COVERAGE_USER_HIT / COVERAGE_USER

    """

    def __init__(self):
        super(HIT_RATE, self).__init__()
        self.cumulative_HR = 0.0
        self.n_users = 0

    def add_recommendations(self, is_relevant):
        self.cumulative_HR += np.any(is_relevant)
        self.n_users += 1

    def get_metric_value(self):
        if self.n_users == 0:
            return 0.0

        return self.cumulative_HR/self.n_users

    def merge_with_other(self, other_metric_object):
        assert other_metric_object is HIT_RATE, "HR: attempting to merge with a metric object of different type"

        self.cumulative_HR += other_metric_object.cumulative_HR
        self.n_users += other_metric_object.n_users




def arhr_all_hits(is_relevant):
    # average reciprocal hit-rank (ARHR) of all relevant items
    # As opposed to MRR, ARHR takes into account all relevant items and not just the first
    # pag 17
    # http://glaros.dtc.umn.edu/gkhome/fetch/papers/itemrsTOIS04.pdf
    # https://emunix.emich.edu/~sverdlik/COSC562/ItemBasedTopTen.pdf

    p_reciprocal = 1/np.arange(1,len(is_relevant)+1, 1.0, dtype=np.float64)
    arhr_score = is_relevant.dot(p_reciprocal)

    assert not np.isnan(arhr_score), "ARHR_all_hits is NaN"
    return arhr_score


def precision(is_relevant):

    if len(is_relevant) == 0:
        precision_score = 0.0
    else:
        precision_score = np.sum(is_relevant, dtype=np.float64) / len(is_relevant)

    assert 0 <= precision_score <= 1, precision_score
    return precision_score


def precision_recall_min_denominator(is_relevant, n_test_items):

    if len(is_relevant) == 0:
        precision_score = 0.0
    else:
        precision_score = np.sum(is_relevant, dtype=np.float64) / min(n_test_items, len(is_relevant))

    assert 0 <= precision_score <= 1, precision_score
    return precision_score



def recall(is_relevant, pos_items):

    recall_score = np.sum(is_relevant, dtype=np.float64) / pos_items.shape[0]

    assert 0 <= recall_score <= 1, recall_score
    return recall_score




def ndcg(ranked_list, pos_items, relevance=None, at=None):

    if relevance is None:
        relevance = np.ones_like(pos_items)
    assert len(relevance) == pos_items.shape[0]

    # Create a dictionary associating item_id to its relevance
    # it2rel[item] -> relevance[item]
    it2rel = {it: r for it, r in zip(pos_items, relevance)}

    # Creates array of length "at" with the relevance associated to the item in that position
    rank_scores = np.asarray([it2rel.get(it, 0.0) for it in ranked_list[:at]], dtype=np.float64)

    # DCG uses the relevance of the recommended items
    rank_dcg = dcg(rank_scores)

    if rank_dcg == 0.0:
        return 0.0

    # IDCG has all relevances to 1 (or the values provided), up to the number of items in the test set that can fit in the list length
    ideal_dcg = dcg(np.sort(relevance)[::-1][:at])

    if ideal_dcg == 0.0:
        return 0.0

    ndcg_ = rank_dcg / ideal_dcg

    return ndcg_


def dcg(scores):
    return np.sum(np.divide(np.power(2, scores) - 1, np.log2(np.arange(scores.shape[0], dtype=np.float64) + 2)),
                  dtype=np.float64)




####################################################################################################################
###############                 BEYOND-ACCURACY METRICS
####################################################################################################################


class _Global_Item_Distribution_Counter(_Metrics_Object):
    """
    This abstract class implements the basic functions to calculate the global distribution of items
    recommended by the algorithm and is used by various diversity metrics
    """
    def __init__(self, n_items, ignore_items):
        super(_Global_Item_Distribution_Counter, self).__init__()

        self.recommended_counter = np.zeros(n_items, dtype=np.float)
        self.ignore_items = ignore_items.astype(np.int).copy()


    def add_recommendations(self, recommended_items_ids):
        if len(recommended_items_ids) > 0:
            self.recommended_counter[recommended_items_ids] += 1

    def _get_recommended_items_counter(self):

        recommended_counter = self.recommended_counter.copy()

        recommended_counter_mask = np.ones_like(recommended_counter, dtype = np.bool)
        recommended_counter_mask[self.ignore_items] = False

        recommended_counter = recommended_counter[recommended_counter_mask]

        return recommended_counter


    def merge_with_other(self, other_metric_object):
        assert isinstance(other_metric_object, self.__class__), "{}: attempting to merge with a metric object of different type".format(self.__class__)

        self.recommended_counter += other_metric_object.recommended_counter

    def get_metric_value(self):
        raise NotImplementedError()




class Coverage_Item(_Global_Item_Distribution_Counter):
    """
    Item coverage represents the percentage of the overall items which were recommended
    https://gab41.lab41.org/recommender-systems-its-not-all-about-the-accuracy-562c7dceeaff
    """

    def __init__(self, n_items, ignore_items):
        super(Coverage_Item, self).__init__( n_items, ignore_items)

    def get_metric_value(self):

        recommended_mask = self._get_recommended_items_counter() > 0

        return recommended_mask.sum()/len(recommended_mask)




class Coverage_Item_HIT(_Global_Item_Distribution_Counter):
    """
    Item coverage represents the percentage of the overall test items which were correctly recommended
    https://gab41.lab41.org/recommender-systems-its-not-all-about-the-accuracy-562c7dceeaff
    """

    def __init__(self, n_items, ignore_items):
        super(Coverage_Item_HIT, self).__init__(n_items, ignore_items)

    def add_recommendations(self, recommended_items_ids, is_relevant):
        super(Coverage_Item_HIT, self).add_recommendations(np.array(recommended_items_ids)[is_relevant])

    def get_metric_value(self):

        recommended_mask = self._get_recommended_items_counter() > 0

        return recommended_mask.sum()/len(recommended_mask)




class Items_In_GT(_Metrics_Object):
    """
    Items_In_GT represents the percentage of the overall items that have at least an interaction in the GT
    """

    def __init__(self, URM_test, ignore_items):
        super(Items_In_GT, self).__init__()

        URM_test.eliminate_zeros()

        self.interaction_in_GT_counter = np.ediff1d(sps.csc_matrix(URM_test).indptr)
        self.ignore_items = ignore_items.astype(np.int).copy()

    def add_recommendations(self, recommended_items_ids):
        pass

    def get_metric_value(self):

        in_GT_mask = self.interaction_in_GT_counter > 0
        in_GT_mask[self.ignore_items] = False

        return in_GT_mask.sum()/(len(in_GT_mask) - len(self.ignore_items))



class Users_In_GT(_Metrics_Object):
    """
    Users_In_GT represents the percentage of the overall users that have at least an interaction in the GT
    """

    def __init__(self, URM_test, ignore_users):
        super(Users_In_GT, self).__init__()

        URM_test.eliminate_zeros()

        self.interaction_in_GT_counter = np.ediff1d(sps.csr_matrix(URM_test).indptr)
        self.ignore_users = ignore_users.astype(np.int).copy()

    def add_recommendations(self, recommended_items_ids):
        pass

    def get_metric_value(self):

        in_GT_mask = self.interaction_in_GT_counter > 0
        in_GT_mask[self.ignore_users] = False

        return in_GT_mask.sum()/(len(in_GT_mask) - len(self.ignore_users))




class Coverage_User(_Metrics_Object):
    """
    User coverage represents the percentage of the overall users for which we can make recommendations.
    If there is at least one recommendation the user is considered as covered

    The users *not* covered include:
    - Users for which I am not computing the recommendations (i.e., they have no test data)
    - Users for which the model does not succeed in generating recommendations

    https://gab41.lab41.org/recommender-systems-its-not-all-about-the-accuracy-562c7dceeaff
    """

    def __init__(self, n_users, ignore_users):
        super(Coverage_User, self).__init__()
        self.users_mask = np.zeros(n_users, dtype=np.bool)
        self.n_ignore_users = len(ignore_users)

    def add_recommendations(self, recommended_items_ids, user_id):
        self.users_mask[user_id] = len(recommended_items_ids)>0

    def get_metric_value(self):
        return self.users_mask.sum()/(len(self.users_mask)-self.n_ignore_users)

    def merge_with_other(self, other_metric_object):
        assert other_metric_object is Coverage_User, "Coverage_User: attempting to merge with a metric object of different type"

        self.users_mask = np.logical_or(self.users_mask, other_metric_object.users_mask)



class Coverage_User_HIT(_Metrics_Object):
    """
    User coverage represents the percentage of the overall users for which we can make at least one correct recommendation.
    If there is at least one correct recommendation the user is considered as covered.

    Note that this is different w.r.t. HR, HR uses as denominator only the number of users for which a recommendation was computed.
    Therefore COVERAGE_USER_HIT = HR * COVERAGE_USER

    https://gab41.lab41.org/recommender-systems-its-not-all-about-the-accuracy-562c7dceeaff
    """

    def __init__(self, n_users, ignore_users):
        super(Coverage_User_HIT, self).__init__()
        self.users_mask = np.zeros(n_users, dtype=np.bool)
        self.n_ignore_users = len(ignore_users)

    def add_recommendations(self, is_relevant, user_id):
        self.users_mask[user_id] = np.any(is_relevant)

    def get_metric_value(self):
        return self.users_mask.sum()/(len(self.users_mask)-self.n_ignore_users)

    def merge_with_other(self, other_metric_object):
        assert other_metric_object is Coverage_User, "Coverage_User_HIT: attempting to merge with a metric object of different type"

        self.users_mask = np.logical_or(self.users_mask, other_metric_object.users_mask)




class Gini_Diversity(_Global_Item_Distribution_Counter):
    """
    Gini diversity index, computed from the Gini Index but with inverted range, such that high values mean higher diversity
    This implementation ignores zero-occurrence items

    # From https://github.com/oliviaguest/gini
    # based on bottom eq: http://www.statsdirect.com/help/content/image/stat0206_wmf.gif
    # from: http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
    #
    # http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.459.8174&rep=rep1&type=pdf
    """

    def __init__(self, n_items, ignore_items):
        super(Gini_Diversity, self).__init__(n_items, ignore_items)

    def get_metric_value(self):

        recommended_counter = self._get_recommended_items_counter()
        gini_diversity = _compute_diversity_gini(recommended_counter)

        return gini_diversity




def _compute_diversity_gini(recommended_counter):
    """
    The function computes the gini diversity of the given recommended item distribution.
    This is NOT the Gini index, rather a variation of it such that high values mean higher diversity
    :param recommended_counter:
    :return:
    """

    n_items = len(recommended_counter)

    recommended_counter_sorted = np.sort(recommended_counter)       # values must be sorted
    index = np.arange(1, n_items+1)                                 # index per array element

    #gini_index = (np.sum((2 * index - n_items  - 1) * recommended_counter_sorted)) / (n_items * np.sum(recommended_counter_sorted))
    gini_diversity = 2*np.sum((n_items + 1 - index)/(n_items+1) * recommended_counter_sorted/np.sum(recommended_counter_sorted))

    return gini_diversity




class Diversity_Herfindahl(_Global_Item_Distribution_Counter):
    """
    The Herfindahl index is also known as Concentration index, it is used in economy to determine whether the market quotas
    are such that an excessive concentration exists. It is here used as a diversity index, if high means high diversity.

    It is known to have a small value range in recommender systems, between 0.9 and 1.0

    The Herfindahl index is a function of the square of the probability an item has been recommended to any user, hence
    The Herfindahl index is equivalent to MeanInterList diversity as they measure the same quantity.

    # http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.459.8174&rep=rep1&type=pdf
    """

    def __init__(self, n_items, ignore_items):
        super(Diversity_Herfindahl, self).__init__(n_items, ignore_items)

    def get_metric_value(self):

        recommended_counter = self._get_recommended_items_counter()
        herfindahl_index = _compute_diversity_herfindahl(recommended_counter)

        return herfindahl_index



def _compute_diversity_herfindahl(recommended_counter):

    if recommended_counter.sum() != 0:
        herfindahl_index = 1 - np.sum((recommended_counter / recommended_counter.sum()) ** 2)
    else:
        herfindahl_index = np.nan

    return herfindahl_index





class Shannon_Entropy(_Global_Item_Distribution_Counter):
    """
    Shannon Entropy is a well known metric to measure the amount of information of a certain string of data.
    Here is applied to the global number of times an item has been recommended.

    It has a lower bound and can reach values over 12.0 for random recommenders.
    A high entropy means that the distribution is random uniform across all users.

    Note that while a random uniform distribution
    (hence all items with SIMILAR number of occurrences)
    will be highly diverse and have high entropy, a perfectly uniform distribution
    (hence all items with EXACTLY IDENTICAL number of occurrences)
    will have 0.0 entropy while being the most diverse possible.

    """

    def __init__(self, n_items, ignore_items):
        super(Shannon_Entropy, self).__init__(n_items, ignore_items)

    def get_metric_value(self):

        recommended_counter = self._get_recommended_items_counter()
        shannon_entropy = _compute_shannon_entropy(recommended_counter)

        return shannon_entropy





def _compute_shannon_entropy(recommended_counter):

    # Ignore from the computation both ignored items and items with zero occurrence.
    # Zero occurrence items will have zero probability and will not change the result, butt will generate nans if used in the log
    recommended_counter_mask = np.ones_like(recommended_counter, dtype = np.bool)
    recommended_counter_mask[recommended_counter == 0] = False

    recommended_counter = recommended_counter[recommended_counter_mask]

    n_recommendations = recommended_counter.sum()

    recommended_probability = recommended_counter/n_recommendations

    shannon_entropy = -np.sum(recommended_probability * np.log2(recommended_probability))

    return shannon_entropy







class Novelty(_Metrics_Object):
    """
    Novelty measures how "novel" a recommendation is in terms of how popular the item was in the train set.

    Due to this definition, the novelty of a cold item (i.e. with no interactions in the train set) is not defined,
    in this implementation cold items are ignored and their contribution to the novelty is 0.

    A recommender with high novelty will be able to recommend also long queue (i.e. unpopular) items.

    Mean self-information  (Zhou 2010)
    """

    def __init__(self, URM_train):
        super(Novelty, self).__init__()

        URM_train = sps.csc_matrix(URM_train)
        URM_train.eliminate_zeros()
        self.item_popularity = np.ediff1d(URM_train.indptr)

        self.novelty = 0.0
        self.n_evaluated_users = 0
        self.n_items = len(self.item_popularity)
        self.n_interactions = self.item_popularity.sum()


    def add_recommendations(self, recommended_items_ids):

        self.n_evaluated_users += 1

        if len(recommended_items_ids)>0:
            recommended_items_popularity = self.item_popularity[recommended_items_ids]

            probability = recommended_items_popularity/self.n_interactions
            probability = probability[probability!=0]

            self.novelty += np.sum(-np.log2(probability)/self.n_items)


    def get_metric_value(self):

        if self.n_evaluated_users == 0:
            return 0.0

        return self.novelty/self.n_evaluated_users

    def merge_with_other(self, other_metric_object):
        assert other_metric_object is Novelty, "Novelty: attempting to merge with a metric object of different type"

        self.novelty = self.novelty + other_metric_object.novelty
        self.n_evaluated_users = self.n_evaluated_users + other_metric_object.n_evaluated_users





class AveragePopularity(_Metrics_Object):
    """
    Average popularity the recommended items have in the train data.
    The popularity is normalized by setting as 1 the item with the highest popularity in the train data
    """

    def __init__(self, URM_train):
        super(AveragePopularity, self).__init__()

        URM_train = sps.csc_matrix(URM_train)
        URM_train.eliminate_zeros()
        item_popularity = np.ediff1d(URM_train.indptr)


        self.cumulative_popularity = 0.0
        self.n_evaluated_users = 0
        self.n_items = URM_train.shape[0]
        self.n_interactions = item_popularity.sum()

        self.item_popularity_normalized = item_popularity/item_popularity.max()


    def add_recommendations(self, recommended_items_ids):

        self.n_evaluated_users += 1

        if len(recommended_items_ids)>0:
            recommended_items_popularity = self.item_popularity_normalized[recommended_items_ids]

            self.cumulative_popularity += np.sum(recommended_items_popularity)/len(recommended_items_ids)


    def get_metric_value(self):

        if self.n_evaluated_users == 0:
            return 0.0

        return self.cumulative_popularity/self.n_evaluated_users

    def merge_with_other(self, other_metric_object):
        assert other_metric_object is Novelty, "AveragePopularity: attempting to merge with a metric object of different type"

        self.cumulative_popularity = self.cumulative_popularity + other_metric_object.cumulative_popularity
        self.n_evaluated_users = self.n_evaluated_users + other_metric_object.n_evaluated_users





class Diversity_similarity(_Metrics_Object):
    """
    Intra list diversity computes the diversity of items appearing in the recommendations received by each single user, by using an item_diversity_matrix.

    It can be used, for example, to compute the diversity in terms of features for a collaborative recommender.

    A content-based recommender will have low IntraList diversity if that is computed on the same features the recommender uses.
    A TopPopular recommender may exhibit high IntraList diversity.

    """

    def __init__(self, item_diversity_matrix):
        super(Diversity_similarity, self).__init__()

        assert np.all(item_diversity_matrix >= 0.0) and np.all(item_diversity_matrix <= 1.0), \
            "item_diversity_matrix contains value greated than 1.0 or lower than 0.0"

        self.item_diversity_matrix = item_diversity_matrix

        self.n_evaluated_users = 0
        self.diversity = 0.0


    def add_recommendations(self, recommended_items_ids):

        current_recommended_items_diversity = 0.0

        for item_index in range(len(recommended_items_ids)-1):

            item_id = recommended_items_ids[item_index]

            item_other_diversity = self.item_diversity_matrix[item_id, recommended_items_ids]
            item_other_diversity[item_index] = 0.0

            current_recommended_items_diversity += np.sum(item_other_diversity)


        self.diversity += current_recommended_items_diversity/(len(recommended_items_ids)*(len(recommended_items_ids)-1))

        self.n_evaluated_users += 1


    def get_metric_value(self):

        if self.n_evaluated_users == 0:
            return 0.0

        return self.diversity/self.n_evaluated_users

    def merge_with_other(self, other_metric_object):
        assert other_metric_object is Diversity_similarity, "Diversity: attempting to merge with a metric object of different type"

        self.diversity = self.diversity + other_metric_object.diversity
        self.n_evaluated_users = self.n_evaluated_users + other_metric_object.n_evaluated_users





class Diversity_MeanInterList(_Metrics_Object):
    """
    MeanInterList diversity measures the uniqueness of different users' recommendation lists.

    It can be used to measure how "diversified" are the recommendations different users receive.

    While the original proposal called this metric "Personalization", we do not use this name since the highest MeanInterList diversity
    is exhibited by a non personalized Random recommender.

    pag. 3, http://www.pnas.org/content/pnas/107/10/4511.full.pdf
    @article{zhou2010solving,
      title={Solving the apparent diversity-accuracy dilemma of recommender systems},
      author={Zhou, Tao and Kuscsik, Zolt{\'a}n and Liu, Jian-Guo and Medo, Mat{\'u}{\v{s}} and Wakeling, Joseph Rushton and Zhang, Yi-Cheng},
      journal={Proceedings of the National Academy of Sciences},
      volume={107},
      number={10},
      pages={4511--4515},
      year={2010},
      publisher={National Acad Sciences}
    }


    It was demonstrated by Ferrari Dacrema that this metric does not require to compute the common items all possible
    couples of users have in common but rather it is only sensitive to the total number of times each item has been recommended.
    MeanInterList diversity is a function of the square of the probability an item has been recommended to any user, hence
    MeanInterList diversity is equivalent to the Herfindahl index as they measure the same quantity.
    See
    @inproceedings{DBLP:conf/aaai/Dacrema21,
      author    = {Maurizio {Ferrari Dacrema}},
      title     = {Demonstrating the Equivalence of List Based and Aggregate Metrics
                   to Measure the Diversity of Recommendations (Student Abstract)},
      booktitle = {Thirty-Fifth {AAAI} Conference on Artificial Intelligence, {AAAI}
                   2021, Thirty-Third Conference on Innovative Applications of Artificial
                   Intelligence, {IAAI} 2021, The Eleventh Symposium on Educational Advances
                   in Artificial Intelligence, {EAAI} 2021, Virtual Event, February 2-9,
                   2021},
      pages     = {15779--15780},
      publisher = {{AAAI} Press},
      year      = {2021},
      url       = {https://ojs.aaai.org/index.php/AAAI/article/view/17886},
    }

    A TopPopular recommender that does not remove seen items will have 0.0 MeanInterList diversity.

    # The formula is diversity_cumulative += 1 - common_recommendations(user1, user2)/cutoff
    # for each couple of users, except the diagonal. It is VERY computationally expensive
    # We can move the 1 and cutoff outside of the summation. Remember to exclude the diagonal
    # co_counts = URM_predicted.dot(URM_predicted.T)
    # co_counts[np.arange(0, n_user, dtype=np.int):np.arange(0, n_user, dtype=np.int)] = 0
    # diversity = (n_user**2 - n_user) - co_counts.sum()/self.cutoff

    # If we represent the summation of co_counts separating it for each item, we will have:
    # co_counts.sum() = co_counts_item1.sum()  + co_counts_item2.sum() ...
    # If we know how many times an item has been recommended, co_counts_item1.sum() can be computed as how many couples of
    # users have item1 in common. If item1 has been recommended n times, the number of couples is n*(n-1)
    # Therefore we can compute co_counts.sum() value as:
    # np.sum(np.multiply(item-occurrence, item-occurrence-1))

    # The naive implementation URM_predicted.dot(URM_predicted.T) might require an hour of computation
    # The last implementation has a negligible computational time even for very big datasets

    """

    def __init__(self, n_items, cutoff):
        super(Diversity_MeanInterList, self).__init__()

        self.recommended_counter = np.zeros(n_items, dtype=np.float64)

        self.n_evaluated_users = 0
        self.n_items = n_items
        self.diversity = 0.0
        self.cutoff = cutoff


    def add_recommendations(self, recommended_items_ids):

        assert len(recommended_items_ids) <= self.cutoff, "Diversity_MeanInterList: recommended list is contains more elements than cutoff"

        self.n_evaluated_users += 1

        if len(recommended_items_ids) > 0:
            self.recommended_counter[recommended_items_ids] += 1




    def get_metric_value(self):

        # Requires to compute the number of common elements for all couples of users
        if self.n_evaluated_users == 0:
            return 1.0

        cooccurrences_cumulative = np.sum(self.recommended_counter**2) - self.n_evaluated_users*self.cutoff

        # All user combinations except diagonal
        all_user_couples_count = self.n_evaluated_users**2 - self.n_evaluated_users

        diversity_cumulative = all_user_couples_count - cooccurrences_cumulative/self.cutoff

        self.diversity = diversity_cumulative/all_user_couples_count

        return self.diversity


    def get_theoretical_max(self):

        global_co_occurrence_count = (self.n_evaluated_users*self.cutoff)**2/self.n_items - self.n_evaluated_users*self.cutoff

        mild = 1 - 1/(self.n_evaluated_users**2 - self.n_evaluated_users)*(global_co_occurrence_count/self.cutoff)

        return mild

    def merge_with_other(self, other_metric_object):

        assert other_metric_object is Diversity_MeanInterList, "Diversity_MeanInterList: attempting to merge with a metric object of different type"

        assert np.all(self.recommended_counter >= 0.0), "Diversity_MeanInterList: self.recommended_counter contains negative counts"
        assert np.all(other_metric_object.recommended_counter >= 0.0), "Diversity_MeanInterList: other.recommended_counter contains negative counts"

        self.recommended_counter += other_metric_object.recommended_counter
        self.n_evaluated_users += other_metric_object.n_evaluated_users



####################################################################################################################
###############                 TESTS
####################################################################################################################

metrics = ['AUC', 'Precision' 'Recall', 'MAP', 'NDCG']


def pp_metrics(metric_names, metric_values, metric_at):
    """
    Pretty-prints metric values
    :param metrics_arr:
    :return:
    """
    assert len(metric_names) == len(metric_values)
    if isinstance(metric_at, int):
        metric_at = [metric_at] * len(metric_values)
    return ' '.join(['{}: {:.4f}'.format(mname, mvalue) if mcutoff is None or mcutoff == 0 else
                     '{}@{}: {:.4f}'.format(mname, mcutoff, mvalue)
                     for mname, mcutoff, mvalue in zip(metric_names, metric_at, metric_values)])


class TestRecall(unittest.TestCase):
    def runTest(self):
        pos_items = np.asarray([2, 4, 5, 10])
        ranked_list_1 = np.asarray([1, 2, 3, 4, 5])
        ranked_list_2 = np.asarray([10, 5, 2, 4, 3])
        ranked_list_3 = np.asarray([1, 3, 6, 7, 8])
        self.assertTrue(np.allclose(recall(ranked_list_1, pos_items), 3. / 4))
        self.assertTrue(np.allclose(recall(ranked_list_2, pos_items), 1.0))
        self.assertTrue(np.allclose(recall(ranked_list_3, pos_items), 0.0))

        thresholds = [1, 2, 3, 4, 5]
        values = [0.0, 1. / 4, 1. / 4, 2. / 4, 3. / 4]
        for at, val in zip(thresholds, values):
            self.assertTrue(np.allclose(np.asarray(recall(ranked_list_1, pos_items, at=at)), val))


class TestPrecision(unittest.TestCase):
    def runTest(self):
        pos_items = np.asarray([2, 4, 5, 10])
        ranked_list_1 = np.asarray([1, 2, 3, 4, 5])
        ranked_list_2 = np.asarray([10, 5, 2, 4, 3])
        ranked_list_3 = np.asarray([1, 3, 6, 7, 8])
        self.assertTrue(np.allclose(precision(ranked_list_1, pos_items), 3. / 5))
        self.assertTrue(np.allclose(precision(ranked_list_2, pos_items), 4. / 5))
        self.assertTrue(np.allclose(precision(ranked_list_3, pos_items), 0.0))

        thresholds = [1, 2, 3, 4, 5]
        values = [0.0, 1. / 2, 1. / 3, 2. / 4, 3. / 5]
        for at, val in zip(thresholds, values):
            self.assertTrue(np.allclose(np.asarray(precision(ranked_list_1, pos_items, at=at)), val))


class TestRR(unittest.TestCase):
    def runTest(self):
        pos_items = np.asarray([2, 4, 5, 10])
        ranked_list_1 = np.asarray([1, 2, 3, 4, 5])
        ranked_list_2 = np.asarray([10, 5, 2, 4, 3])
        ranked_list_3 = np.asarray([1, 3, 6, 7, 8])
        self.assertTrue(np.allclose(rr(ranked_list_1, pos_items), 1. / 2))
        self.assertTrue(np.allclose(rr(ranked_list_2, pos_items), 1.))
        self.assertTrue(np.allclose(rr(ranked_list_3, pos_items), 0.0))

        thresholds = [1, 2, 3, 4, 5]
        values = [0.0, 1. / 2, 1. / 2, 1. / 2, 1. / 2]
        for at, val in zip(thresholds, values):
            self.assertTrue(np.allclose(np.asarray(rr(ranked_list_1, pos_items, at=at)), val))


class TestMAP(unittest.TestCase):
    def runTest(self):
        pos_items = np.asarray([2, 4, 5, 10])
        ranked_list_1 = np.asarray([1, 2, 3, 4, 5])
        ranked_list_2 = np.asarray([10, 5, 2, 4, 3])
        ranked_list_3 = np.asarray([1, 3, 6, 7, 8])
        ranked_list_4 = np.asarray([11, 12, 13, 14, 15, 16, 2, 4, 5, 10])
        ranked_list_5 = np.asarray([2, 11, 12, 13, 14, 15, 4, 5, 10, 16])
        self.assertTrue(np.allclose(map(ranked_list_1, pos_items), (1. / 2 + 2. / 4 + 3. / 5) / 4))
        self.assertTrue(np.allclose(map(ranked_list_2, pos_items), 1.0))
        self.assertTrue(np.allclose(map(ranked_list_3, pos_items), 0.0))
        self.assertTrue(np.allclose(map(ranked_list_4, pos_items), (1. / 7 + 2. / 8 + 3. / 9 + 4. / 10) / 4))
        self.assertTrue(np.allclose(map(ranked_list_5, pos_items), (1. + 2. / 7 + 3. / 8 + 4. / 9) / 4))

        thresholds = [1, 2, 3, 4, 5]
        values = [
            0.0,
            1. / 2 / 2,
            1. / 2 / 3,
            (1. / 2 + 2. / 4) / 4,
            (1. / 2 + 2. / 4 + 3. / 5) / 4
        ]
        for at, val in zip(thresholds, values):
            self.assertTrue(np.allclose(np.asarray(map(ranked_list_1, pos_items, at)), val))


class TestNDCG(unittest.TestCase):
    def runTest(self):
        pos_items = np.asarray([2, 4, 5, 10])
        pos_relevances = np.asarray([5, 4, 3, 2])
        ranked_list_1 = np.asarray([1, 2, 3, 4, 5])  # rel = 0, 5, 0, 4, 3
        ranked_list_2 = np.asarray([10, 5, 2, 4, 3])  # rel = 2, 3, 5, 4, 0
        ranked_list_3 = np.asarray([1, 3, 6, 7, 8])  # rel = 0, 0, 0, 0, 0
        idcg = ((2 ** 5 - 1) / np.log(2) +
                (2 ** 4 - 1) / np.log(3) +
                (2 ** 3 - 1) / np.log(4) +
                (2 ** 2 - 1) / np.log(5))
        self.assertTrue(np.allclose(dcg(np.sort(pos_relevances)[::-1]), idcg))
        self.assertTrue(np.allclose(ndcg(ranked_list_1, pos_items, pos_relevances),
                                    ((2 ** 5 - 1) / np.log(3) +
                                     (2 ** 4 - 1) / np.log(5) +
                                     (2 ** 3 - 1) / np.log(6)) / idcg))
        self.assertTrue(np.allclose(ndcg(ranked_list_2, pos_items, pos_relevances),
                                    ((2 ** 2 - 1) / np.log(2) +
                                     (2 ** 3 - 1) / np.log(3) +
                                     (2 ** 5 - 1) / np.log(4) +
                                     (2 ** 4 - 1) / np.log(5)) / idcg))
        self.assertTrue(np.allclose(ndcg(ranked_list_3, pos_items, pos_relevances), 0.0))


if __name__ == '__main__':
    unittest.main()
