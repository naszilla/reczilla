import os
import shutil
import time
import logging
import shlex
import numpy as np
import random
import tensorflow as tf


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
