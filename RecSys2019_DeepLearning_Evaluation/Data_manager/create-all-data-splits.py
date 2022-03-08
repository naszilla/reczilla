"""
Creates all splits for datasets in dataset_handler.DATASET_READER_LIST.

"""

import argparse
import os
from posixpath import split

from numpy import require
from dataset_handler import DATASET_READER_LIST
from Data_manager.DataSplitter_leave_k_out import DataSplitter_leave_k_out

ALL_SPLITTERS = [
    (DataSplitter_leave_k_out, {
        "k_out_value": 1, 
        "forbid_new_split": False, 
        "force_new_split": False, 
        "use_validation_set": True
    }),
    ]

def create_all_splits(data_dir, splits_dir):
    for idx, reader in enumerate(DATASET_READER_LIST):
        data_folder = os.path.join(data_dir, reader.__name__) + "/"
        print(f"attempting to load dataset: {reader.__name__} from directory {data_folder}")

        for splitter, s_kwargs in ALL_SPLITTERS:
            save_split_path = os.path.join(splits_dir, reader.DATASET_SUBFOLDER, splitter.DATA_SPLITTER_NAME)
            try:
                # attempt to load the dataset, unless it is already downloaded
                data_reader = reader(reload_from_original_data="as-needed", verbose=False, folder=data_folder)
                loaded_dataset = data_reader.load_data()
                data_splitter = splitter(data_reader, **s_kwargs, folder=save_split_path, verbose=False)
                data_splitter.load_data()
                print(f"SUCCESS - Saved split of {reader.__name__} using splitter {data_splitter.DATA_SPLITTER_NAME}")
            
            except Exception as e:
                print(f"FAILURE - {reader.__name__}: exception raised while loading dataset. skipping this dataset")
                print(f"EXCEPTION: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="directory where the downloaded dataset has been stored. If a dataset is not downloaded, it will be downloaded.")
    parser.add_argument("--splits-dir", required=True, help="directory where the splits will be saved")
    args = parser.parse_args()
    create_all_splits(args.data_dir, args.splits_dir)

