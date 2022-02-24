"""
functions for splitting a loaded and saved dataset

this is a "light" version of the DataSplitter functionality, with a slightly different data format:
- each split is defined by 2+ files, including:
-- URM_test.npz
-- URM_train.npz
-- URM_validation.npz
-- dataset_ICM_mappers.zip (only if ICM is used)
-- dataset_UCM_mappers.zip (only if UCM is used)

all npz files are written by numpy, and all zip files are written by the Base.DataIO functions.
"""
import os
import glob
from scipy.sparse import save_npz, load_npz

from Data_manager.Dataset import Dataset
from Data_manager.split_functions.split_train_validation_leave_k_out import (
    split_train_leave_k_out_user_wise,
)

SPLIT_TYPES = [
    "leave_k_out",
]


def write_split(
    dataset: Dataset, split_type: str, out_directory: str, split_args: dict = {}
):
    """
    split a dataset and write all files to out_directory.
    out_directory must not exist, or it must be an empty directory.
    """

    assert split_type in SPLIT_TYPES, f"split type not recognized: {split_type}"

    # if the output directory doesn't exist, create it
    if not os.path.exists(out_directory):
        os.makedirs(out_directory)
    else:
        # if it does exist, make sure it is an empty directory
        assert os.path.isdir(out_directory), f"out_directory is not a directory: {out_directory}"
        assert len(os.listdir(out_directory)) == 0, f"out_directory is not empty: {out_directory}"

    # objects that we will write
    write_objects = {}

    if split_type == "leave_k_out":

        # split the dataset.
        # the function split_train_leave_k_out_user_wise has different return structure depending on its args
        if "use_validation_set" in split_args:
            assert split_args["use_validation_set"] in [True, False]
            if split_args["use_validation_set"]:
                (
                    write_objects["URM_train"],
                    write_objects["URM_validation"],
                    write_objects["URM_test"],
                ) = split_train_leave_k_out_user_wise(
                    dataset.get_URM_all(), **split_args
                )
            else:
                (
                    write_objects["URM_train"],
                    write_objects["URM_test"],
                ) = split_train_leave_k_out_user_wise(
                    dataset.get_URM_all(), **split_args
                )

    else:
        raise Exception(f"we don't handle this split type: {split_type}")

    # write each object that was defined. different files are written differently
    for x in ["URM_train", "URM_test", "URM_validation"]:
        if x in write_objects:
            save_npz(os.path.join(out_directory, x + ".npz"), write_objects[x], compressed=True)
            print(f"object {x} written to directory {out_directory}")

    # TODO: handle UCM/ICM


def read_split(directory):
    """return a dictionary of all files written by write_split"""

    object_dict = {}
    for x in ["URM_train.npz", "URM_test.npz", "URM_validation.npz"]:
        # look for the file
        files = glob.glob(os.path.join(directory, x))
        if len(files) > 0:
            object_dict[x] = load_npz(files[0])
            print(f"loaded object {x} from file: {files[0]}")

    # TODO: handle UCM/ICM

    return object_dict