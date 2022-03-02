import os
import time
import logging
import shlex


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


def config_to_sequence(config_filepath):
    """
    read a config file and return a sequence of args.
    lines beginning with '#' are ignored.
    each line is stripped and then split by shlex.split, and all lines are concatenated into a list
    """
    sequence = []
    with open(config_filepath, "r") as f:
        for line in f:
            l = line.strip()
            if not l.startswith("#"):
                sequence.extend(shlex.split(l))

    return sequence
