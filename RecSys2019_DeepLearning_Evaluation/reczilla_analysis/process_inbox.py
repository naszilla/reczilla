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

    # make sure that directories we will create do not exist, and that the base dir does exist
    assert base_path.exists(), f"base dir does not exist: {base_path}"

    inbox_path.mkdir(exist_ok=True)

    # pull all files
    os.system(f"gsutil -m rsync gs://reczilla-results/inbox {inbox_path}")
    # os.system(f"gsutil -m cp gs://reczilla-results/inbox/result_07*.zip  {inbox_path}")

    # create a reader object
    dataIO = DataIO(str(inbox_path) + os.sep)

    # merge all dfs
    df_list = []

    print("aggregating data...") ### PRINT
    # for each result matching pattern inbox/*.zip, place it in the appropriate folder
    for result_file in inbox_path.glob("*.zip"):

        try:
            # create a csv with useful results
            result_df = result_to_df(result_file)
            df_list.append(result_df)

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
