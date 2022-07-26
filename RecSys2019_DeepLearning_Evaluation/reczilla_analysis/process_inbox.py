# download all results locally and organize into directory structure dataset/split/alg
import argparse
import os
import pandas as pd
from pathlib import Path
from Base.DataIO import DataIO
from Utils.reczilla_utils import result_to_df


def run(args):

    base_path = Path(args.base_dir)
    inbox_path = base_path.joinpath("inbox")
    result_path = base_path.joinpath("structured_results")

    # make sure that directories we will create do not exist, and that the base dir does exist
    assert base_path.exists(), f"base dir does not exist: {base_path}"

    inbox_path.mkdir(exist_ok=True)
    result_path.mkdir(exist_ok=True)

    # pull all files
    os.system(f"gsutil -m rsync gs://reczilla-results/inbox {inbox_path}")
    # os.system(f"gsutil -m cp gs://reczilla-results/inbox/result_052*.zip  {inbox_path}")

    # create a reader object
    dataIO = DataIO(str(inbox_path) + os.sep)

    # merge all dfs
    df_list = []

    # for each result matching pattern inbox/*.zip, place it in the appropriate folder
    for result_file in inbox_path.glob("*.zip"):

        try:
            # read metadata
            data = dataIO.load_data(result_file.name)

            # fields that will be used to place the file in a structured dir
            alg_name = data["search_params"]["alg_name"]
            dataset_name = data["search_params"]["dataset_name"]
            split_name = data["search_params"]["split_name"]

            new_path = result_path.joinpath(
                dataset_name, split_name, alg_name, result_file.name
            )

            # create directory if it doesn't exist
            new_path.parent.mkdir(parents=True, exist_ok=True)

            # move the file to the appropriate directory
            result_file.rename(new_path)

            # create a csv with useful results
            result_df = result_to_df(new_path)
            df_list.append(result_df)
            result_df.to_csv(str(new_path.parent.joinpath(f"{new_path.stem}.csv")), sep=";", index=False)

        except Exception as e:
            print(f"exception while reading file {result_file}. skipping this file")
            print(f"exception: {e}")

        # if local inbox is empty, delete it:
        if not any(inbox_path.iterdir()):
            inbox_path.rmdir()

    print(f"finished organizing files. now merging csv.")
    df_final = pd.concat(df_list, ignore_index=True)
    results_path = base_path.joinpath("results.csv")
    if results_path.exists():
        print(f"WARNING: overwriting results file {results_path}")
    df_final.to_csv(results_path, index=False, sep=";")
    print("done")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "base_dir",
        type=str,
        help="base directory where files will be downloaded and structured.",
    )

    args = parser.parse_args()

    run(args)
