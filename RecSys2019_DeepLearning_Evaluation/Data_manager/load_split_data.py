"""
functions for loading a split dataset from a directory, regardless of the origin of the dataset.

all datasets are read using a "dummy" dataset reader, Data_manager.DataReader.GenericDataReader

after loading the split, we return the URM, ICM, and UCM splits, returned by DataSplitter_leave_k_out.get_holdout_split.
"""
from Data_manager.DataReader import GenericDataReader
from Data_manager.DataSplitter_leave_k_out import DataSplitter_leave_k_out


def read_leave_k_out_split(split_directory):
    """
    return the URM, ICM, and UCM dictionaries.
    """
    dataSplitter = DataSplitter_leave_k_out(GenericDataReader())
    dataSplitter.load_data(save_folder_path=split_directory)
    return dataSplitter.SPLIT_URM_DICT, dataSplitter.SPLIT_ICM_DICT, dataSplitter.SPLIT_UCM_DICT
