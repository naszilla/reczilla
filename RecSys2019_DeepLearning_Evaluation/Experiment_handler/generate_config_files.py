import argparse
from pathlib import Path

from Experiment_handler.Experiment import Experiment
from algorithm_handler import ALGORITHM_NAME_LIST, algorithm_handler
from dataset_handler import DATASET_READER_NAME_LIST

SPLIT_TYPE_LIST = [
    "DataSplitter_leave_k_out"
]

def run(args):
    print("CREATING CONFIG FILES IN DIRECTORY:")
    print(f"OUTPUT_DIR: {Path(args.output_dir).resolve()}")
    print(f"CONFIG FILE DIR: {args.experiment_name}")
    experiment = Experiment(Path(args.output_dir), None, args.experiment_name)
    # create a config file for each alg + dataset combination. for now, only include leave_k_out splitter
    for alg_name in ALGORITHM_NAME_LIST:

        # get the maximum number of points for this alg
        _, _, _, max_points = algorithm_handler(alg_name)
        num_samples = min(max_points, args.num_samples)
        for dataset_name in DATASET_READER_NAME_LIST:
            for split_type in SPLIT_TYPE_LIST:
                kwargs = {
                    "write-zip": "",  # always write a zip file
                    "experiment-name": args.experiment_name,
                    "alg-seed": 1,
                    "num-samples": num_samples,
                    "result-dir": args.result_dir,
                    "split-dir": args.split_dir,
                }

                file_name = "config.txt"
                experiment.prepare_config(file_name, args.data_dir, dataset_name, split_type, alg_name, **kwargs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output-dir",
        type=str,
        help="directory where config file structure will be written. this directory should exist",
        required=True,
    )
    parser.add_argument(
        "--config-directory-name",
        type=str,
        help="name of the directory that will be created for config files.",
        required=True,
    )

    # args that will be written to config files

    parser.add_argument(
        "--data-dir",
        type=str,
        help="full path of the data directory.",
        default=None,
        required=False,
    )
    parser.add_argument(
        "--split-dir",
        type=str,
        help="full path of the directory containing a single split.",
        default="",
        required=False,
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        help="name of the experiment (in the config file).",
        required=True,
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        help="number of parameter samples.",
        required=True,
    )
    parser.add_argument(
        "--result-dir",
        type=str,
        help="path of the directory where experiments will be written. this directory should exist.",
        required=True,
    )
    args = parser.parse_args()

    run(args)