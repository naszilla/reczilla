# download all results locally and update the old ones
import argparse
import os
from pathlib import Path
from Base.DataIO import DataIO

# run the following command to pull all results:
# gsutil -m rsync gs://reczilla-results/inbox ./inbox


def run(args):

    base_path = Path(args.base_dir)
    inbox_path = base_path.joinpath("inbox")
    result_path = base_path.joinpath("structured_results")

    # make sure that directories we will create do not exist, and that the base dir does exist
    assert base_path.exists(), f"base dir does not exist: {base_path}"
    assert not inbox_path.exists(), f"inbox dir already exists: {inbox_path}"
    assert not result_path.exists(), f"result dir already exists: {result_path}"

    inbox_path.mkdir()
    result_path.mkdir()

    # pull all files
    os.system(f"gsutil -m rsync gs://reczilla-results/inbox {inbox_path}")

    # create a reader object
    dataIO = DataIO(str(inbox_path) + os.sep)

    # for each result matching pattern inbox/*.zip, place it in the appropriate folder
    for result_file in inbox_path.glob("*.zip"):

        try:
            # read metadata
            data = dataIO.load_data(result_file.name)

            # fields that will be used to place the file in a structured dir
            alg_name = data["search_params"]["alg_name"]
            dataset_name = data["search_params"]["dataset_name"]
            split_name = data["search_params"]["split_name"]

            # print(
            #     f"discovered result for ALG: {alg_name}, SPLIT: {split_name}, DATASET: {dataset_name}"
            # )

            new_path = result_path.joinpath(
                dataset_name, split_name, alg_name, result_file.name
            )

            # create directory if it doesn't exist
            assert not new_path.exists(), f"file already exists at path: {new_path}"
            new_path.parent.mkdir(parents=True, exist_ok=True)

            # move the file to the appropriate directory
            result_file.rename(new_path)

        except Exception as e:
            print(f"exception while reading file {result_file}. skipping this file")
            print(f"exception: {e}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "base_dir",
        type=str,
        help="base directory where files will be downloaded and structured.",
    )

    args = parser.parse_args()

    run(args)
