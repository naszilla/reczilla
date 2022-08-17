import os
import shutil
import time
import logging

import numpy as np
import random

import pandas as pd
import tensorflow as tf

from Base.DataIO import DataIO

from algorithm_handler import algorithm_handler

TIME_FORMAT = "%Y%m%d_%H%M%S"


LOG_FORMAT = "[%(asctime)-15s] [%(filename)s:%(funcName)s] : %(message)s"


def get_logger(logfile=None):
    logger = logging.getLogger("experiment_logs")
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(LOG_FORMAT)
    if logfile is not None:
        fh = logging.FileHandler(logfile)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    else:
        logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    return logger


def time_to_str(time_format):
    # return a string representation of the current time
    return time.strftime(time_format)


def str_to_time(x, time_format):
    # inverse of time_to_str(): return a datetime object
    return time.strptime(x, time_format)


def make_archive(source, destination):
    """
    a helper function because shutil.make_archive is too confusing on its own. adapted from:
    http://www.seanbehan.com/how-to-use-python-shutil-make_archive-to-zip-up-a-directory-recursively-including-the-root-folder/

    zip the folder at "source" and write it to the file at "destination". the file type is read from arg "destination"
    """

    base = os.path.basename(destination)
    name = base.split(".")[0]
    format = base.split(".")[1]
    archive_from = os.path.dirname(source)
    archive_to = os.path.basename(source.strip(os.sep))
    shutil.make_archive(name, format, archive_from, archive_to)
    shutil.move("%s.%s" % (name, format), destination)


def set_deterministic(seed):
    """
    Set the seeds for all used libraries and enables deterministic behavior
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # Tensorflow Determinism
    # See https://github.com/NVIDIA/framework-determinism
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
    os.environ["HOROVOD_FUSION_THRESHOLD"] = "0"  # Determinism for multiple GPUs


def result_to_df(result_zip_path):
    """
    create a df with one row for each set of hyperparameters, and one col for each metric
    """

    # load metadata structure
    dataIO = DataIO(str(result_zip_path.parent) + os.sep)
    data = dataIO.load_data(result_zip_path.name)

    # the search parameter num_samples is the max number of samples we'll find. some algorithms provide
    # fewer samples (smaller hyperparam space) so the actual number of samples is just the length of this list.
    num_samples = len(data["hyperparameters_list"])

    use_validation_set = "result_on_validation_list" in data

    # make sure that each of the lists has the correct length
    assert (
        len(data["result_on_test_list"]) == num_samples
    ), f"test metric list has len = {len(data['result_on_test_list'])}. expected {num_samples}"
    if use_validation_set:
        assert (
            len(data["result_on_validation_list"]) == num_samples
        ), f"validatino metric list has len = {len(data['result_on_validation_list'])}. expected {num_samples}"
    assert (
        len(data["time_on_test_list"]) == num_samples
    ), f"time-on-test list has len = {len(data['time_on_test_list'])}. expected {num_samples}"
    assert (
        len(data["time_on_validation_list"]) == num_samples
    ), f"time-on-val list has len = {len(data['time_on_validation_list'])}. expected {num_samples}"
    assert (
        len(data["time_on_train_list"]) == num_samples
    ), f"time-on-train list has len = {len(data['time_on_train_list'])}. expected {num_samples}"
    assert (
        len(data["exception_list"]) == num_samples
    ), f"exception list has len = {len(data['exception_list'])}. expected {num_samples}"

    # store each row in a dict. some of these rows have common values: store these in a template
    row_template = data["search_params"]
    row_list = []

    # create one row for each of the hyperparameter samples
    for i in range(num_samples):

        row_dict = row_template.copy()
        row_dict["sample_number"] = i

        # add hyperparameters to the row
        if type(data["hyperparameters_list"][i]) is dict:
            # if not, there was an error with this result, so skip it. all hyperparams will be NA in the final dataframe
            for key, val in data["hyperparameters_list"][i].items():
                row_dict[f"param_{key}"] = val

        # add test results
        if type(data["result_on_test_list"][i]) is dict:
            # if not, there was an error with this result, so skip it.
            for key, val in data["result_on_test_list"][i].items():
                # each val should be a dict with different cutoffs
                for subkey, subval in val.items():
                    # each subkey/value should be a metric name/value
                    row_dict[f"test_metric_{subkey}_cut_{key}"] = subval

        if use_validation_set:
            if data["result_on_validation_list"][i] is dict:
                # if not, there was an error with this result, so skip it.
                for key, val in data["result_on_test_list"][i].items():
                    # each val should be a dict with different cutoffs
                    for subkey, subval in val.items():
                        # each subkey/value should be a metric name/value
                        row_dict[f"val_metric_{subkey}_cut_{key}"] = subval

        # basic metrics
        row_dict["time_on_val"] = data["time_on_validation_list"][i]
        row_dict["time_on_test"] = data["time_on_test_list"][i]
        row_dict["time_on_train"] = data["time_on_train_list"][i]
        row_dict["exception"] = data["exception_list"][i]
        if "hyperparameters_source" in data:
            row_dict["hyperparameters_source"] = data["hyperparameters_source"][i]

        row_list.append(row_dict)

    # create a df
    df = pd.DataFrame(row_list)

    return df


def get_parameterized_alg(alg_param_string, param_seed=3):
    """
    given a string in the format "<alg name>:<hyperparameter source>", return the algorithm class and hyperparameters

    these strings can be found in the meta-datasets produced by notebooks/prepare_metadataset_v#.ipynb,
    in column "alg_param_name"
    """

    # split the string into alg and hyperparameter source
    split_str = alg_param_string.split(":")
    assert (
        len(split_str) == 2
    ), f"alg_param_string must have format '<alg name>:<hyperparameter source>'. alg_param_string = {alg_param_string}"
    alg_name = split_str[0]
    hyperparam_source = split_str[1]

    # if alg_name contains KNN, we handle it differently..
    if "KNN" in alg_name:
        split_hparam_string = hyperparam_source.split("_")
        knn_sim_type = split_hparam_string[0]
        knn_param_source = "_".join(split_hparam_string[1:])

        alg_name = alg_name + "_" + knn_sim_type
        hyperparam_source = knn_param_source

    # retrieve algorithm class and parameter space
    (
        alg_class,
        parameter_search_space,
        search_input_recommender_args,
        max_points,
    ) = algorithm_handler(alg_name)

    # parse hyperparams source string and generate hyperparams
    if hyperparam_source == "default":
        hparams = parameter_search_space.default
    else:
        assert (
            hyperparam_source[: len("random_")] == "random_"
        ), f"hyperparam source stirng not recognized: {hyperparam_source}"
        sample_number = int(hyperparam_source[len("random_") :])  # zero-indexed

        # this code is adapted from RandomSearch.search()
        use_default_params = True  # this is the default in RandomSearch.search()
        hyperparam_rs = np.random.RandomState(param_seed)

        # sample random hyperparam values. if we're using the default param set, take (n_samples - 1) samples
        n_random_samples = sample_number + 1
        assert n_random_samples > 0
        hyperparam_samples = parameter_search_space.random_samples(
            n_random_samples, rs=hyperparam_rs,
        )
        assert len(hyperparam_samples) == n_random_samples
        hparams = hyperparam_samples[-1]

    ###### DEBUGGING ######
    # test code
    # import pickle
    # metadata_file = '/Users/duncan/research/active_projects/reczilla/RecSys2019_DeepLearning_Evaluation/metadatasets/metadata-v2.pkl'
    # with open(metadata_file, "rb") as f:
    #     meta_dataset = pickle.load(f)
    ### get a random alg string
    # alg_param_string = meta_dataset["alg_param_name"].sample().values[0]
    # ## parse it
    # alg, hp, search = get_parameterized_alg(alg_param_string, param_seed=3)
    # print(alg_param_string)
    # print(f"alg: {alg}")
    # print(f"hyperparams: {hp}")
    # ## check the original params in meta-dataset
    # check_row = meta_dataset.loc[meta_dataset["alg_param_name"] == alg_param_string, :].iloc[0]
    # print("params:")
    # print({param: check_row["param_" + param] for param in hp.keys()})

    return alg_class, hparams, search_input_recommender_args
