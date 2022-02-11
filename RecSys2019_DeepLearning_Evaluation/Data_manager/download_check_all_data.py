"""
Download all datasets in dataset_handler.DATASET_READER_LIST.

if a dataset is found, don't redownload it. just validate that it exists
"""

from dataset_handler import DATASET_READER_LIST
import os
import argparse


def run(args):
    """
    only take one argument: args.data_dir. download each dataset to args.data_dir/<data reader name>
    """

    for i, reader_class in enumerate(DATASET_READER_LIST):
        data_folder = os.path.join(args.data_dir, reader_class.__name__) + "/"
        print(
            f"attempting to download dataset: {reader_class.__name__} to directory {data_folder}"
        )

        # attempt to load the dataset, unless it is already downloaded
        data_reader = reader_class(reload_from_original_data="as-needed")
        loaded_dataset = data_reader.load_data(save_folder_path=data_folder)

    # make sure that URM_all is available and has positive dimensions
    URM_all = loaded_dataset.get_URM_all()

    assert URM_all.shape[0] > 0, f"URM_all does not have nonzero dimension (0)"
    assert URM_all.shape[1] > 0, f"URM_all does not have nonzero dimension (1)"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data-dir", help="directory where data will be downloaded")
    args = parser.parse_args()
    run(args)
