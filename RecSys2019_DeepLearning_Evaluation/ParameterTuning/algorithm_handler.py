#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 1 Feb 2022

@author: Duncan C McElfresh

code for iterating over the set of recommender algorithms
- list ALGORITHM_NAME_LIST contains all valid algorithm names
- function algorithm_handler takes an algorithm name and returns the algorithm class, parameter
    search space, and args passed to the constructor and init functions
"""

from skopt.space import Real, Integer, Categorical
from ParameterTuning.SearchAbstractClass import SearchInputRecommenderArgs
from ParameterTuning.ParameterSpace import ParameterSpace


######################################################################
##########                                                  ##########
##########                  USER-INDEPENDENT                ##########
##########                                                  ##########
######################################################################
from Base.NonPersonalizedRecommender import TopPop, Random, GlobalEffects

######################################################################
##########                                                  ##########
##########                  PURE COLLABORATIVE              ##########
##########                                                  ##########
######################################################################
# KNN
from KNN.UserKNNCFRecommender import UserKNNCFRecommender
from KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from GraphBased.P3alphaRecommender import P3alphaRecommender
from GraphBased.RP3betaRecommender import RP3betaRecommender
from EASE_R.EASE_R_Recommender import EASE_R_Recommender

# KNN machine learning
from SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from SLIM_ElasticNet.SLIMElasticNetRecommender import SLIMElasticNetRecommender

# Matrix Factorization
from MatrixFactorization.PureSVDRecommender import PureSVDRecommender
from MatrixFactorization.IALSRecommender import IALSRecommender
from MatrixFactorization.NMFRecommender import NMFRecommender
from MatrixFactorization.Cython.MatrixFactorization_Cython import (
    MatrixFactorization_BPR_Cython,
    MatrixFactorization_FunkSVD_Cython,
    MatrixFactorization_AsySVD_Cython,
)
from Conferences.IJCAI.DELF_our_interface.DELFWrapper import DELF_MLP_RecommenderWrapper, DELF_EF_RecommenderWrapper

######################################################################
##########                                                  ##########
##########              NEURAL NETWORK METHODS              ##########
##########                                                  ##########
######################################################################

from Conferences.IJCAI.NeuRec_our_interface.UNeuRecWrapper import (
    UNeuRec_RecommenderWrapper,
)
from Conferences.IJCAI.NeuRec_our_interface.INeuRecWrapper import (
    INeuRec_RecommenderWrapper,
)
from Conferences.RecSys.SpectralCF_our_interface.SpectralCF_RecommenderWrapper import (
    SpectralCF_RecommenderWrapper,
)
from Conferences.WWW.MultiVAE_our_interface.MultiVAE_RecommenderWrapper import (
    Mult_VAE_RecommenderWrapper,
)

ALGORITHM_NAME_LIST = [
    "ItemKNNCF_asymmetric",
    "ItemKNNCF_tversky",
    "ItemKNNCF_euclidean",
    "ItemKNNCF_cosine",
    "ItemKNNCF_jaccard",
    "ItemKNNCF_dice",
    "UserKNNCF_asymmetric",
    "UserKNNCF_tversky",
    "UserKNNCF_euclidean",
    "UserKNNCF_cosine",
    "UserKNNCF_jaccard",
    "UserKNNCF_dice",
    "TopPop",
    "GlobalEffects",
    "Random",
    "P3alphaRecommender",
    "RP3betaRecommender",
    "MatrixFactorization_FunkSVD_Cython",
    "MatrixFactorization_AsySVD_Cython",
    "MatrixFactorization_BPR_Cython",
    "IALSRecommender",
    "PureSVDRecommender",
    "NMFRecommender",
    "SLIM_BPR_Cython",
    "SLIMElasticNetRecommender",
    "EASE_R_Recommender",
    "INeuRec_RecommenderWrapper",  # see run_IJCAI_18_NeuRec.py
    "UNeuRec_RecommenderWrapper",  # see run_IJCAI_18_NeuRec.py
    "SpectralCF_RecommenderWrapper",  # see run_RecSys_18_SpectralCF.py
    "Mult_VAE_RecommenderWrapper",  # see run_WWW_18_Mult_VAE.py
    "DELF_MLP_RecommenderWrapper",  # see run_IJCAI_17_DELF.py
    "DELF_EF_RecommenderWrapper",  # see run_IJCAI_17_DELF.py
]

BASE_KNN_ARGS = {
    "topK": Integer(5, 1000),
    "shrink": Integer(0, 1000),
    "normalize": Categorical([True, False]),
}


def algorithm_handler(algorithm_name):
    """
    Returns:
        - alg: handle of algorithm class
        - space: dict of search space
        - search_input_recommender_args (SearchInputRecommenderArgs): fixed arguments passed to algorithm init and fit

    Each space is a dict: keys are names (string) of the hyperparameter, and values are of type
    skopt.space.{Real | Integer | Categorical}.

    Only "collaborative" recsys methods are implemented here (only using URM, not item/user features)

    """

    assert (
        algorithm_name in ALGORITHM_NAME_LIST
    ), f"Algorithm {algorithm_name} not recognized."

    # ------------------------------------
    # define SearchInputRecommenderArgs
    # ------------------------------------

    # this is just an empty container. since some algorithms need additional arguments during init or fit,
    # those args can be passed here.
    # NOTE: the training dataset needs to be added to this object before using it
    # NOTE: early stopping kwargs also need to be added to this object, only for certain algorithms (these args use
    # evaluation objects, so we don't initialize them here.
    search_input_recommender_args = SearchInputRecommenderArgs()

    # ------------------------------------
    # define algorithm class
    # ------------------------------------

    if algorithm_name.startswith("ItemKNNCF"):
        alg = ItemKNNCFRecommender
    elif algorithm_name.startswith("UserKNNCF"):
        alg = UserKNNCFRecommender
    else:
        # parse the recommender class using its name
        alg = globals()[algorithm_name]

    # ------------------------------------
    # define param spaces
    # ------------------------------------

    space = {}

    # maximum number of points to sample. -1 = no max
    max_points = 1e10

    # for all KNN algorithms
    # in the original codebase, the constant params were included in the search space. here, we instead add them to
    # the search_input_recommender_args object.
    if algorithm_name.endswith("KNNCF_asymmetric"):
        space.update(BASE_KNN_ARGS)
        space["asymmetric_alpha"] = Real(low=0, high=2, prior="uniform")
        # remove normalize - this is a constant for this model
        del space["normalize"]
        search_input_recommender_args.FIT_KEYWORD_ARGS["normalize"] = True
        search_input_recommender_args.FIT_KEYWORD_ARGS["similarity"] = "asymmetric"

    if algorithm_name.endswith("KNNCF_tversky"):
        space.update(BASE_KNN_ARGS)
        space.update(
            {
                "tversky_alpha": Real(low=0, high=2, prior="uniform"),
                "tversky_beta": Real(low=0, high=2, prior="uniform"),
            }
        )
        # remove normalize - this is a constant for this model
        del space["normalize"]
        search_input_recommender_args.FIT_KEYWORD_ARGS["normalize"] = True
        search_input_recommender_args.FIT_KEYWORD_ARGS["similarity"] = "tversky"

    if algorithm_name.endswith("KNNCF_euclidean"):
        space.update(BASE_KNN_ARGS)
        space.update(
            {
                "normalize_avg_row": Categorical([True, False]),
                "similarity_from_distance_mode": Categorical(["lin", "log", "exp"]),
            }
        )
        search_input_recommender_args.FIT_KEYWORD_ARGS["similarity"] = "euclidean"

    if algorithm_name.endswith("KNNCF_cosine"):
        space.update(BASE_KNN_ARGS)
        space["feature_weighting"] = Categorical(["none", "BM25", "TF-IDF"])
        search_input_recommender_args.FIT_KEYWORD_ARGS["similarity"] = "cosine"

    if algorithm_name.endswith("KNNCF_jaccard"):
        space.update(BASE_KNN_ARGS)
        search_input_recommender_args.FIT_KEYWORD_ARGS["similarity"] = "jaccard"

    if algorithm_name.endswith("KNNCF_dice"):
        space.update(BASE_KNN_ARGS)
        search_input_recommender_args.FIT_KEYWORD_ARGS["similarity"] = "dice"

    # no params needed. only allow one sample
    if any([alg is c for c in [TopPop, GlobalEffects, Random]]):
        max_points = 1

    # other algs

    if alg is P3alphaRecommender:
        space = {
            "topK": Integer(5, 1000),
            "alpha": Real(low=0, high=2, prior="uniform"),
            "normalize_similarity": Categorical([True, False]),
        }

    if alg is RP3betaRecommender:
        space = {
            "topK": Integer(5, 1000),
            "alpha": Real(low=0, high=2, prior="uniform"),
            "beta": Real(low=0, high=2, prior="uniform"),
            "normalize_similarity": Categorical([True, False]),
        }

    if alg is MatrixFactorization_FunkSVD_Cython:
        space = {
            "sgd_mode": Categorical(["sgd", "adagrad", "adam"]),
            "use_bias": Categorical([True, False]),
            "batch_size": Categorical([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]),
            "num_factors": Integer(1, 200),
            "item_reg": Real(low=1e-5, high=1e-2, prior="log-uniform"),
            "user_reg": Real(low=1e-5, high=1e-2, prior="log-uniform"),
            "learning_rate": Real(low=1e-4, high=1e-1, prior="log-uniform"),
            "negative_interactions_quota": Real(low=0.0, high=0.5, prior="uniform"),
        }
        search_input_recommender_args.FIT_KEYWORD_ARGS["epochs"] = 500

    if alg is MatrixFactorization_AsySVD_Cython:
        space = {
            "sgd_mode": Categorical(["sgd", "adagrad", "adam"]),
            "use_bias": Categorical([True, False]),
            "num_factors": Integer(1, 200),
            "item_reg": Real(low=1e-5, high=1e-2, prior="log-uniform"),
            "user_reg": Real(low=1e-5, high=1e-2, prior="log-uniform"),
            "learning_rate": Real(low=1e-4, high=1e-1, prior="log-uniform"),
            "negative_interactions_quota": Real(low=0.0, high=0.5, prior="uniform"),
        }
        search_input_recommender_args.FIT_KEYWORD_ARGS["epochs"] = 500
        search_input_recommender_args.FIT_KEYWORD_ARGS["batch_size"] = 1

    if alg is MatrixFactorization_BPR_Cython:
        space = {
            "sgd_mode": Categorical(["sgd", "adagrad", "adam"]),
            "num_factors": Integer(1, 200),
            "batch_size": Categorical([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]),
            "positive_reg": Real(low=1e-5, high=1e-2, prior="log-uniform"),
            "negative_reg": Real(low=1e-5, high=1e-2, prior="log-uniform"),
            "learning_rate": Real(low=1e-4, high=1e-1, prior="log-uniform"),
        }
        search_input_recommender_args.FIT_KEYWORD_ARGS["epochs"] = 1500
        search_input_recommender_args.FIT_KEYWORD_ARGS["positive_threshold_BPR"] = None

    if alg is IALSRecommender:
        space = {
            "num_factors": Integer(1, 200),
            "confidence_scaling": Categorical(["linear", "log"]),
            "alpha": Real(low=1e-3, high=50.0, prior="log-uniform"),
            "epsilon": Real(low=1e-3, high=10.0, prior="log-uniform"),
            "reg": Real(low=1e-5, high=1e-2, prior="log-uniform"),
        }
        search_input_recommender_args.FIT_KEYWORD_ARGS["epochs"] = 300

    if alg is PureSVDRecommender:
        space = {
            "num_factors": Integer(1, 350),
        }
        max_points = 300

    if alg is NMFRecommender:
        space = {
            "num_factors": Integer(1, 350),
            "solver": Categorical(["coordinate_descent", "multiplicative_update"]),
            "init_type": Categorical(["random", "nndsvda"]),
            "beta_loss": Categorical(["frobenius", "kullback-leibler"]),
        }
        max_points = 2800

    if alg is SLIM_BPR_Cython:
        space = {
            "topK": Integer(5, 1000),
            "symmetric": Categorical([True, False]),
            "sgd_mode": Categorical(["sgd", "adagrad", "adam"]),
            "lambda_i": Real(low=1e-5, high=1e-2, prior="log-uniform"),
            "lambda_j": Real(low=1e-5, high=1e-2, prior="log-uniform"),
            "learning_rate": Real(low=1e-4, high=1e-1, prior="log-uniform"),
        }
        search_input_recommender_args.FIT_KEYWORD_ARGS["epochs"] = 1500
        search_input_recommender_args.FIT_KEYWORD_ARGS["positive_threshold_BPR"] = None
        search_input_recommender_args.FIT_KEYWORD_ARGS[
            "train_with_sparse_weights"
        ] = None

    if alg is SLIMElasticNetRecommender:
        space = {
            "topK": Integer(5, 1000),
            "l1_ratio": Real(low=1e-5, high=1.0, prior="log-uniform"),
            "alpha": Real(low=1e-3, high=1.0, prior="uniform"),
        }

    if alg is EASE_R_Recommender:
        space = {
            "l2_norm": Real(low=1e0, high=1e7, prior="log-uniform"),
        }
        max_points = 1000
        search_input_recommender_args.FIT_KEYWORD_ARGS["topK"] = None
        search_input_recommender_args.FIT_KEYWORD_ARGS["normalize_matrix"] = False

    if alg is UNeuRec_RecommenderWrapper or alg is INeuRec_RecommenderWrapper:
        # TODO: make sure this is a reasonable parameter space
        space = {
            "num_neurons": Integer(
                3, 500
            ),  # number of neurons in the first four layers
            "num_factors": Integer(2, 100),  # number of neurons in the last two layers
            "dropout_percentage": Real(low=0.0, high=0.3),
            "learning_rate": Real(low=1e-6, high=1e-1, prior="log-uniform"),
            "regularization_rate": Real(low=1e-4, high=1e1, prior="log-uniform"),
        }
        search_input_recommender_args.FIT_KEYWORD_ARGS["epochs"] = 100
        search_input_recommender_args.FIT_KEYWORD_ARGS["batch_size"] = 1024

    if alg is SpectralCF_RecommenderWrapper:
        # TODO: make sure this is a reasonable parameter space
        space = {
            "batch_size": Categorical([128, 256, 512, 1024, 2048]),
            "embedding_size": Categorical([4, 8, 16, 32]),
            "decay": Real(low=1e-5, high=1e-1, prior="log-uniform"),
            "learning_rate": Real(low=1e-5, high=1e-2, prior="log-uniform"),
            "k": Integer(low=1, high=6),
        }
        search_input_recommender_args.FIT_KEYWORD_ARGS["epochs"] = 1000

    if alg is Mult_VAE_RecommenderWrapper:
        # TODO: make sure this is a reasonable parameter space
        space = {
            "total_anneal_steps": Integer(1000, 1000000),
            "lam": Real(
                1e-10, 1e-1, prior="log-uniform"
            ),  # strength of l2 regularization
            "lr": Real(1e-6, 1e-2, prior="log-uniform"),  # learning rate
        }

        search_input_recommender_args.FIT_KEYWORD_ARGS[
            "p_dims"
        ] = None  # TODO: this uses default. define a reasonable parameter range
        # search_input_recommender_args.FIT_KEYWORD_ARGS["q_dims"] = None  # TODO: the fit function does not currently take q_dims as an arg
        search_input_recommender_args.FIT_KEYWORD_ARGS["epochs"] = 1000
        search_input_recommender_args.FIT_KEYWORD_ARGS["batch_size"] = 500

    if alg is DELF_EF_RecommenderWrapper or alg is DELF_MLP_RecommenderWrapper:
        # TODO: make sure this is a reasonable parameter space. see DELFWrapper._DELF_RecommenderWrapper.fit()
        # TODO: we should understand what these parameters are, at a high level...
        num_factors = 64
        space = {
         'learning_rate': Real(1e-6, 1e-2, prior="log-uniform"),
         'num_negatives': Integer(3, 4),  # TODO: not sure what this is
         }
        search_input_recommender_args.FIT_KEYWORD_ARGS["learner"] = "adam"
        search_input_recommender_args.FIT_KEYWORD_ARGS["verbose"] = False
        search_input_recommender_args.FIT_KEYWORD_ARGS["layers"] = (num_factors * 4, num_factors * 2, num_factors)
        search_input_recommender_args.FIT_KEYWORD_ARGS["regularization_layers"] = (0, 0, 0)
        search_input_recommender_args.FIT_KEYWORD_ARGS["epochs"] = 500
        search_input_recommender_args.FIT_KEYWORD_ARGS["batch_size"] = 256

    return alg, ParameterSpace(space), search_input_recommender_args, max_points
