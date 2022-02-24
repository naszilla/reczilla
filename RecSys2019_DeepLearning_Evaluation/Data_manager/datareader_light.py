"""
function for reading a dataset that has already been loaded/saved by a DataReader object

this is a "light" implementation of the DataReader functionality
"""
import os

from Data_manager.Dataset import Dataset


def datareader_light(directory):
    """
    attempt to read a dataset from a directory using Dataset.load_data

    this loading function attempts to read one or more of the following files:
    - `dataset_global_attributes.zip`
    - `dataset_URM.zip`
    - `dataset_ICM_mappers.zip`
    - `dataset_UCM_mappers.zip`
    - `dataset_additional_mappers.zip`

    a Dataset object is returned
    """

    assert os.path.exists(directory), f"directory does not exist: {directory}"
    assert os.path.isdir(directory), f"object is not a directory: {directory}"


    # directory paths are expected to end with a path separator. this is required by the codebase
    # TODO: change the codebase to not require the path separator at the end of all directory paths
    assert directory.endswith(os.path.sep), f"directory does not end with {os.path.sep}: {directory}"

    dataset = Dataset()

    try:
        dataset.load_data(directory)
    except Exception as e:
        print(f"exception raised while loading dataset. directory={directory}")
        raise e

    try:
        # verify consistency. this raises an exception if the consistency test fails
        dataset.verify_data_consistency()
    except Exception as e:
        print(f"exception raised while verifying consistency. directory={directory}")
        raise e

    return dataset