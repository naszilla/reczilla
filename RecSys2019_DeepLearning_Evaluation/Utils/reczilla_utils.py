import os
import shutil
import string
import time
import logging
import zipfile

import numpy as np
import random

import pandas as pd
import tensorflow as tf

from Base.DataIO import DataIO

TIME_FORMAT = "%Y%m%d_%H%M%S"


def generate_filepath(output_dir, name, extension):
    # generate filepath, of the format <name>_YYYYMMDD_HHMMDD.<extension>
    timestr = time.strftime("%Y%m%d_%H%M%S")
    output_string = (name + "_%s." + extension) % timestr
    return os.path.join(output_dir, output_string)


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
    tf.random.set_random_seed(seed)

    # Tensorflow Determinism
    # See https://github.com/NVIDIA/framework-determinism
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
    os.environ["HOROVOD_FUSION_THRESHOLD"] = "0"  # Determinism for multiple GPUs

def convert_old_result_file(zip_file_path):
    """
    convert the "old" result zip file format into the new format.

    a new zip file will be created in the same directory as zip_file_path, with the suffix _UPDATED

    this function will raise a TypeError if the zip file is not in the old format
    """

    # create temp dir
    temp_dir = zip_file_path.parent.joinpath(
        "TEMP_"
        + time_to_str(TIME_FORMAT)
        + "_"
        + "".join(random.choices(string.ascii_uppercase + string.digits, k=3))
    )

    # extract the zip to the temp dir
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)

    # find all results, i.e. files that end in '_metadata.zip'
    result_files = [f for f in temp_dir.rglob("*_metadata.zip")]

    if len(result_files) != 1:
        raise TypeError(f"multiple results found in zip archive: {zip_file_path}. we expect only one.")

    result_file = result_files[0]

    # gather the name of the alg, the split, and the dataset from the result path
    alg_name = result_file.parent.name
    split_name = result_file.parent.parent.name
    dataset_name = result_file.parent.parent.parent.name
    experiment_name = result_file.parent.parent.parent.parent.name

    # # in the new base: make the result directory if it doesn't exist, and move the result zip there
    #     new_home = new_base_path.joinpath(dataset_name, split_name, alg_name)
    # new_home.mkdir(parents=True, exist_ok=True)
    # result_file = result_file.rename(new_home.joinpath(result_file.name))

    # read the alg seed, param seed, and timestamp from the zip file
    file_name_split = result_file.name.split("_")
    assert file_name_split[0][:7] == 'algseed'
    alg_seed = int(file_name_split[0][7:])
    assert file_name_split[1][:9] == 'paramseed'
    param_seed = int(file_name_split[1][9:])

    time_str = file_name_split[2] + "_" + file_name_split[3]

    # read the metadata dict from the result zip
    dataIO = DataIO(folder_path=str(result_file.parent) + os.sep)
    data_dict = dataIO.load_data(file_name=result_file.name)

    try:
        cutoff_list = list(data_dict["result_on_test_list"][0].keys())
    except:
        # if the above command failed, the experimnt probably failed
        cutoff_list = None

    data_dict["search_params"] = {
        "time": time_str,
        "dataset_name": dataset_name,
        "split_name": split_name,
        "alg_name": alg_name,
        "num_samples": len(data_dict["hyperparameters_list"]),
        "alg_seed": alg_seed,
        "param_seed": param_seed,
        "cutoff_list": cutoff_list,
        "experiment_name": experiment_name,
    }
    # finally, remove the temp directory
    shutil.rmtree(str(temp_dir))

    # write the new object. use a new dataio object to write to a new location
    dataIO_update = DataIO(folder_path=str(zip_file_path.parent) + os.sep)
    dataIO_update.save_data(file_name=zip_file_path.stem + "_UPDATED", data_dict_to_save=data_dict)

    print(f"wrote updated metadata to {zip_file_path.parent.joinpath(result_file.stem + '_UPDATED')}")


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
    assert len(data["result_on_test_list"]) == num_samples, f"test metric list has len = {len(data['result_on_test_list'])}. expected {num_samples}"
    if use_validation_set:
        assert len(data["result_on_validation_list"]) == num_samples, f"validatino metric list has len = {len(data['result_on_validation_list'])}. expected {num_samples}"
    assert len(data["time_on_test_list"]) == num_samples, f"time-on-test list has len = {len(data['time_on_test_list'])}. expected {num_samples}"
    assert len(data["time_on_validation_list"]) == num_samples, f"time-on-val list has len = {len(data['time_on_validation_list'])}. expected {num_samples}"
    assert len(data["time_on_train_list"]) == num_samples, f"time-on-train list has len = {len(data['time_on_train_list'])}. expected {num_samples}"
    assert len(data["exception_list"]) == num_samples, f"exception list has len = {len(data['exception_list'])}. expected {num_samples}"

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

        row_list.append(row_dict)

    # create a df
    df = pd.DataFrame(row_list)

    return df
