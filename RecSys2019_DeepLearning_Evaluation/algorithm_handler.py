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

# Surprise Algorithms
from SurpriseAlgorithms.Wrappers import CoClustering, SlopeOne

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
from Conferences.IJCAI.DELF_our_interface.DELFWrapper import (
    DELF_MLP_RecommenderWrapper,
    DELF_EF_RecommenderWrapper,
)
from Conferences.IJCAI.ConvNCF_our_interface.ConvNCF_wrapper import (
    ConvNCF_RecommenderWrapper,
)
from Conferences.IJCAI.ConvNCF_our_interface.MFBPR_Wrapper import MFBPR_Wrapper

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
    # "ConvNCF_RecommenderWrapper",  # see run_IJCAI_18_ConvNCF.py  # TODO: there are some bugs in this implementation.
    "MFBPR_Wrapper",  # see run_IJCAI_18_ConvNCF_CNN_embedding.py
    "CoClustering",
    "SlopeOne",
]

BASE_KNN_ARGS = {
    "topK": Integer(5, 1000),
    "shrink": Integer(0, 1000),
    "normalize": Categorical([True, False]),
}

DEFAULT_KNN_ARGS = {"topK": 5, "shrink": 50, "normalize": True}

DEFAULT_NUM_FACTORS = 10
DEFAULT_ITEM_REG = 1e-4
DEFAULT_USER_REG = 1e-3
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_SGD_MODE = "sgd"
DEFAULT_EPOCHS = 500


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

    # container for args that are always passed to fit() function
    fit_keyword_args = {}

    # ------------------------------------
    # knn only - define algorithm class & args
    # ------------------------------------
    KNN_ALG = "KNNCF" in algorithm_name
    if KNN_ALG:
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
    default = {}

    # maximum number of points to sample.
    max_points = 1e10

    # ---- for all KNN algorithms ----
    # in the original codebase, the constant params were included in the search space. here, we instead add them to
    # the search_input_recommender_args object.
    if KNN_ALG:
        space.update(BASE_KNN_ARGS)
        default.update(DEFAULT_KNN_ARGS)
        if algorithm_name.endswith("asymmetric"):
            space["asymmetric_alpha"] = Real(low=0, high=2, prior="uniform")
            default["asymmetric_alpha"] = 1.0
            # remove normalize - this is a constant for this model
            del space["normalize"]
            del default["normalize"]
            fit_keyword_args["normalize"] = True
            fit_keyword_args["similarity"] = "asymmetric"

        elif algorithm_name.endswith("tversky"):
            space.update(
                {
                    "tversky_alpha": Real(low=0, high=2, prior="uniform"),
                    "tversky_beta": Real(low=0, high=2, prior="uniform"),
                }
            )
            default["tversky_alpha"] = 1.0
            default["tversky_beta"] = 1.0

            # remove normalize - this is a constant for this model
            del space["normalize"]
            del default["normalize"]
            fit_keyword_args["normalize"] = True
            fit_keyword_args["similarity"] = "tversky"

        elif algorithm_name.endswith("euclidean"):
            space.update(
                {
                    "normalize_avg_row": Categorical([True, False]),
                    "similarity_from_distance_mode": Categorical(["lin", "log", "exp"]),
                }
            )
            default["normalize_avg_row"] = True
            default["similarity_from_distance_mode"] = "lin"

            fit_keyword_args["similarity"] = "euclidean"

        elif algorithm_name.endswith("cosine"):
            space["feature_weighting"] = Categorical(["none", "BM25", "TF-IDF"])
            default["feature_weighting"] = "none"
            fit_keyword_args["similarity"] = "cosine"

        elif algorithm_name.endswith("jaccard"):
            fit_keyword_args["similarity"] = "jaccard"

        elif algorithm_name.endswith("dice"):
            fit_keyword_args["similarity"] = "dice"
    else:
        # ---- all other (non-KNN) algorithms ----

        # no params needed. only allow one sample
        if any([alg is c for c in [TopPop, GlobalEffects, Random]]):
            max_points = 1

        elif alg is P3alphaRecommender:
            space = {
                "topK": Integer(5, 1000),
                "alpha": Real(low=0, high=2, prior="uniform"),
                "normalize_similarity": Categorical([True, False]),
            }
            default = {
                "topK": 5,
                "alpha": 1.0,
                "normalize_similarity": True,
            }

        elif alg is RP3betaRecommender:
            space = {
                "topK": Integer(5, 1000),
                "alpha": Real(low=0, high=2, prior="uniform"),
                "beta": Real(low=0, high=2, prior="uniform"),
                "normalize_similarity": Categorical([True, False]),
            }
            default = {
                "topK": 5,
                "alpha": 1.0,
                "beta": 1.0,
                "normalize_similarity": True,
            }

        elif alg is MatrixFactorization_FunkSVD_Cython:
            space = {
                "sgd_mode": Categorical(["sgd", "adagrad", "adam"]),
                "use_bias": Categorical([True, False]),
                "batch_size": Categorical(
                    [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
                ),
                "num_factors": Integer(1, 200),
                "item_reg": Real(low=1e-5, high=1e-2, prior="log-uniform"),
                "user_reg": Real(low=1e-5, high=1e-2, prior="log-uniform"),
                "learning_rate": Real(low=1e-4, high=1e-1, prior="log-uniform"),
                "negative_interactions_quota": Real(low=0.0, high=0.5, prior="uniform"),
            }
            default = {
                "sgd_mode": DEFAULT_SGD_MODE,
                "use_bias": True,
                "batch_size": 32,
                "num_factors": DEFAULT_NUM_FACTORS,
                "item_reg": DEFAULT_ITEM_REG,
                "user_reg": DEFAULT_ITEM_REG,
                "learning_rate": DEFAULT_LEARNING_RATE,
                "negative_interactions_quota": 0.2,
            }
            fit_keyword_args["epochs"] = 500

        elif alg is MatrixFactorization_AsySVD_Cython:
            space = {
                "sgd_mode": Categorical(["sgd", "adagrad", "adam"]),
                "use_bias": Categorical([True, False]),
                "num_factors": Integer(1, 200),
                "item_reg": Real(low=1e-5, high=1e-2, prior="log-uniform"),
                "user_reg": Real(low=1e-5, high=1e-2, prior="log-uniform"),
                "learning_rate": Real(low=1e-4, high=1e-1, prior="log-uniform"),
                "negative_interactions_quota": Real(low=0.0, high=0.5, prior="uniform"),
            }
            default = {
                "sgd_mode": DEFAULT_SGD_MODE,
                "use_bias": True,
                "num_factors": DEFAULT_NUM_FACTORS,
                "item_reg": DEFAULT_ITEM_REG,
                "user_reg": DEFAULT_USER_REG,
                "learning_rate": DEFAULT_LEARNING_RATE,
                "negative_interactions_quota": 0.2,
            }
            fit_keyword_args["epochs"] = DEFAULT_EPOCHS
            fit_keyword_args["batch_size"] = 1

        elif alg is MatrixFactorization_BPR_Cython:
            space = {
                "sgd_mode": Categorical(["sgd", "adagrad", "adam"]),
                "num_factors": Integer(1, 200),
                "batch_size": Categorical(
                    [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
                ),
                "positive_reg": Real(low=1e-5, high=1e-2, prior="log-uniform"),
                "negative_reg": Real(low=1e-5, high=1e-2, prior="log-uniform"),
                "learning_rate": Real(low=1e-4, high=1e-1, prior="log-uniform"),
            }
            default = {
                "sgd_mode": DEFAULT_SGD_MODE,
                "num_factors": DEFAULT_NUM_FACTORS,
                "batch_size": 32,
                "positive_reg": 1e-3,
                "negative_reg": 1e-3,
                "learning_rate": DEFAULT_LEARNING_RATE,
            }
            fit_keyword_args["epochs"] = DEFAULT_EPOCHS
            fit_keyword_args["positive_threshold_BPR"] = None

        elif alg is IALSRecommender:
            space = {
                "num_factors": Integer(1, 200),
                "confidence_scaling": Categorical(["linear", "log"]),
                "alpha": Real(low=1e-3, high=50.0, prior="log-uniform"),
                "epsilon": Real(low=1e-3, high=10.0, prior="log-uniform"),
                "reg": Real(low=1e-5, high=1e-2, prior="log-uniform"),
            }
            default = {
                "num_factors": DEFAULT_NUM_FACTORS,
                "confidence_scaling": "linear",
                "alpha": 1.0,
                "epsilon": 1.0,
                "reg": 1e-3,
            }
            fit_keyword_args["epochs"] = DEFAULT_EPOCHS

        elif alg is PureSVDRecommender:
            space = {
                "num_factors": Integer(1, 350),
            }
            default = {
                "num_factors": DEFAULT_NUM_FACTORS,
            }
            max_points = 300

        elif alg is NMFRecommender:
            space = {
                "num_factors": Integer(1, 350),
                "solver": Categorical(["coordinate_descent", "multiplicative_update"]),
                "init_type": Categorical(["random", "nndsvda"]),
                "beta_loss": Categorical(["frobenius", "kullback-leibler"]),
            }
            default = {
                "num_factors": DEFAULT_NUM_FACTORS,
                "solver": "coordinate_descent",
                "init_type": "random",
                "beta_loss": "frobenius",
            }
            max_points = 2800

        elif alg is SLIM_BPR_Cython:
            space = {
                "topK": Integer(5, 1000),
                "symmetric": Categorical([True, False]),
                "sgd_mode": Categorical(["sgd", "adagrad", "adam"]),
                "lambda_i": Real(low=1e-5, high=1e-2, prior="log-uniform"),
                "lambda_j": Real(low=1e-5, high=1e-2, prior="log-uniform"),
                "learning_rate": Real(low=1e-4, high=1e-1, prior="log-uniform"),
            }
            default = {
                "topK": 5,
                "symmetric": True,
                "sgd_mode": DEFAULT_SGD_MODE,
                "lambda_i": 1e-3,
                "lambda_j": 1e-3,
                "learning_rate": DEFAULT_LEARNING_RATE,
            }
            fit_keyword_args["epochs"] = DEFAULT_EPOCHS
            fit_keyword_args["positive_threshold_BPR"] = None
            fit_keyword_args["train_with_sparse_weights"] = None

        elif alg is SLIMElasticNetRecommender:
            space = {
                "topK": Integer(5, 1000),
                "l1_ratio": Real(low=1e-5, high=1.0, prior="log-uniform"),
                "alpha": Real(low=1e-3, high=1e2, prior="uniform"),
            }
            default = {
                "topK": 5,
                "l1_ratio": 0.1,
                "alpha": 1.0,
            }

        elif alg is EASE_R_Recommender:
            space = {
                "l2_norm": Real(low=1e0, high=1e7, prior="log-uniform"),
            }
            default = {
                "l2_norm": 1e3,
            }
            max_points = 1000
            fit_keyword_args["topK"] = None
            fit_keyword_args["normalize_matrix"] = False

        elif alg is UNeuRec_RecommenderWrapper or alg is INeuRec_RecommenderWrapper:
            # TODO: make sure this is a reasonable parameter space
            space = {
                "num_neurons": Integer(
                    3, 500
                ),  # number of neurons in the first four layers
                "num_factors": Integer(
                    2, 100
                ),  # number of neurons in the last two layers
                "dropout_percentage": Real(low=0.0, high=0.3),
                "learning_rate": Real(low=1e-6, high=1e-1, prior="log-uniform"),
                "regularization_rate": Real(low=1e-4, high=1e1, prior="log-uniform"),
            }
            default = {
                "num_neurons": 100,
                "num_factors": DEFAULT_NUM_FACTORS,  # number of neurons in the last two layers
                "dropout_percentage": 0.1,
                "learning_rate": DEFAULT_LEARNING_RATE,
                "regularization_rate": 1e-2,
            }

            fit_keyword_args["epochs"] = DEFAULT_EPOCHS
            fit_keyword_args["batch_size"] = 1024

        elif alg is SpectralCF_RecommenderWrapper:
            # TODO: make sure this is a reasonable parameter space
            space = {
                "batch_size": 256,
                "embedding_size": Categorical([4, 8, 16, 32]),
                "decay": Real(low=1e-5, high=1e-1, prior="log-uniform"),
                "learning_rate": Real(low=1e-5, high=1e-2, prior="log-uniform"),
                "k": Integer(low=1, high=6),
            }
            default = {
                "batch_size": 1024,
                "embedding_size": 16,
                "decay": 0.001,
                "learning_rate": DEFAULT_LEARNING_RATE,
                "k": 3,
            }

            fit_keyword_args["epochs"] = DEFAULT_EPOCHS

        elif alg is Mult_VAE_RecommenderWrapper:
            # TODO: make sure this is a reasonable parameter space
            space = {
                "total_anneal_steps": Integer(1000, 1000000),
                "lam": Real(
                    1e-10, 1e-1, prior="log-uniform"
                ),  # strength of l2 regularization
                "lr": Real(1e-6, 1e-2, prior="log-uniform"),  # learning rate
            }
            default = {
                "total_anneal_steps": 200000,
                "lam": 0.0,
                "lr": DEFAULT_LEARNING_RATE,
            }

            fit_keyword_args[
                "p_dims"
            ] = None  # TODO: this uses default. define a reasonable parameter range
            # fit_keyword_args["q_dims"] = None  # TODO: the fit function does not currently take q_dims as an arg
            fit_keyword_args["epochs"] = DEFAULT_EPOCHS
            fit_keyword_args["batch_size"] = 500

        elif alg is DELF_EF_RecommenderWrapper or alg is DELF_MLP_RecommenderWrapper:
            # TODO: make sure this is a reasonable parameter space. see DELFWrapper._DELF_RecommenderWrapper.fit()
            # TODO: we should understand what these parameters are, at a high level...
            num_factors = 64
            space = {
                "learning_rate": Real(1e-6, 1e-2, prior="log-uniform"),
                "num_negatives": Integer(3, 4),  # TODO: not sure what this is
            }
            default = {
                "learning_rate": DEFAULT_LEARNING_RATE,
                "num_negatives": 4,
            }
            fit_keyword_args["learner"] = "adam"
            fit_keyword_args["verbose"] = False
            fit_keyword_args["layers"] = (
                num_factors * 4,
                num_factors * 2,
                num_factors,
            )
            fit_keyword_args["regularization_layers"] = (
                0,
                0,
                0,
            )
            fit_keyword_args["epochs"] = DEFAULT_EPOCHS
            fit_keyword_args["batch_size"] = 256

        elif alg is ConvNCF_RecommenderWrapper:
            raise NotImplementedError("there are some bugs in the implementation.")
            # space = {
            #     "embedding_size": Categorical([32, 64, 128]),
            #     "hidden_size": Categorical([32, 64, 128]),
            #     "regularization_users_items": Real(1e-4, 1e-1, prior="log-uniform"),
            #     "learning_rate_embeddings": Real(1e-4, 1e-1, prior="log-uniform"),
            #     "learning_rate_CNN": Real(1e-4, 1e-1, prior="log-uniform"),
            # }
            # fit_keyword_args["negative_sample_per_positive"] = 1
            # fit_keyword_args["negative_instances_per_positive"] = 4
            # fit_keyword_args["regularization_weights"] = 10
            # fit_keyword_args["regularization_filter_weights"] = 1
            # fit_keyword_args["channel_size"] = [
            #     32,
            #     32,
            #     32,
            #     32,
            #     32,
            #     32,
            # ]
            # fit_keyword_args["dropout"] = 0.0
            # fit_keyword_args["epoch_verbose"] = 1
            #
            # fit_keyword_args["batch_size"] = 512
            # fit_keyword_args["epochs"] = 1500
            # fit_keyword_args["epochs_MFBPR"] = 500

        elif alg is MFBPR_Wrapper:
            # TODO: make sure this is a reasonable parameter space
            space = {
                "embed_size": Categorical([32, 64, 128]),
                "learning_rate": Real(1e-4, 1e-1, prior="log-uniform"),
            }
            default = {
                "embed_size": 64,
                "learning_rate": DEFAULT_LEARNING_RATE,
            }
            fit_keyword_args["negative_sample_per_positive"] = 1

            fit_keyword_args["batch_size"] = 512
            fit_keyword_args["epochs"] = DEFAULT_EPOCHS
            fit_keyword_args[
                "path_partial_results"
            ] = "./TMP/"  # TODO: we shouldn't be defining paths here.

        elif alg is CoClustering:
            # Based on Autosurprise
            space = {
                "n_cltr_u": Integer(1, 1000),
                "n_cltr_i": Integer(1, 100),
            }
            default = {
                "n_cltr_u": 20,
                "n_cltr_i": 20,
            }

            fit_keyword_args.update({"n_epochs": 20, "random_state": None})

        elif alg is SlopeOne:
            space = {}
            default = {}
            max_points = 1
        else:
            raise Exception(f"algorithm_handler can't handle {algorithm_name}")

    # NOTE: the training dataset needs to be added to this object before using it
    # NOTE: early stopping kwargs also need to be added to this object, only for certain algorithms (these args use
    # evaluation objects, so we don't initialize them here.
    search_input_recommender_args = SearchInputRecommenderArgs(
        FIT_KEYWORD_ARGS=fit_keyword_args
    )

    return (
        alg,
        ParameterSpace(space, default),
        search_input_recommender_args,
        max_points,
    )
