"""
run a single experiment to generate results.

here, a single "experiment" is when we train one recsys algorithm on one train/test/validation split of a dataset, with
one or more sets of hyperparameters for the algorithm. we record the train and test performance for all trained algorithms

for each experiment we regenerate the dataset split. this is reasonable because the dataset splits are O(1mb) in size,
and they are deterministic since we pass random seeds to the split function.
"""
import argparse
from pathlib import Path
from Experiment_handler.Experiment import Experiment


def run(args):

    # False = we don't read from the processed datasets. Instead we read the split data directly
    # for this reason we don't need data_path, which points to the processed data.
    use_processed_data = False

    verbose = False

    # run experiment
    experiment = Experiment(
        Path(args.result_dir),
        args.experiment_name,
        use_processed_data=use_processed_data,
        verbose=verbose,
        log_file=args.log_file,
    )
    experiment.prepare_dataset(args.dataset_name)
    experiment.prepare_split(args.dataset_name, args.split_type, split_path=Path(args.split_dir))
    result_zip = experiment.run_experiment(
        args.dataset_name,
        args.split_type,
        args.alg_name,
        args.num_samples,
        args.alg_seed,
        args.param_seed,
        args.original_split_path,
        result_dir=Path(args.result_dir),
        time_limit=args.time_limit,
    )

    # the result file has a timestamp on it. we will rename this to a generic name, so that our bash script
    # can move the result file to gcloud
    print(f"initial result file: {result_zip}")
    print(f"renaming to: {Path(args.result_dir).joinpath('result.zip')}")
    _ = Path(result_zip).rename(Path(args.result_dir).joinpath("result.zip"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "time_limit",
        type=int,
        help="time limit in seconds",
    )
    parser.add_argument(
        "dataset_name",
        type=str,
        help="name of dataset. we use this to find the dataset and split.",
    )
    parser.add_argument(
        "split_type",
        type=str,
        help="name of datasplitter to use. we use this to find the split directory.",
    )
    parser.add_argument(
        "alg_name", type=str, help="name of the algorithm to use.",
    )
    parser.add_argument(
        "split_dir",
        type=str,
        help="directory containing split data files.",
    )
    parser.add_argument(
        "alg_seed",
        type=int,
        help="random seed passed to the recommender algorithm. only for random algorithms.",
    )
    parser.add_argument(
        "param_seed",
        type=int,
        help="random seed for generating random hyperparameters.",
    )
    parser.add_argument(
        "num_samples",
        type=int,
        help="number of hyperparameter samples.",
    )
    parser.add_argument(
        "result_dir",
        type=str,
        help="directory where result dir structure will be written. this directory should exist.",
    )
    parser.add_argument(
        "experiment_name",
        type=str,
        help="name of the result directory that will be created.",
    )
    parser.add_argument(
        "original_split_path",
        type=str,
        help="full path to the split data. only used for bookkeeping.",
    )
    parser.add_argument(
        "log_file",
        type=str,
        help="full path to a file where logs will be written (appended).",
    )

    args = parser.parse_args()

    run(args)
