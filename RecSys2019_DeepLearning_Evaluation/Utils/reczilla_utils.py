import os
import shutil
import string
import time
import logging
import zipfile

import numpy as np
import random
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

