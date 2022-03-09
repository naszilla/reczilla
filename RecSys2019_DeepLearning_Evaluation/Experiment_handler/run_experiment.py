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
from Utils.reczilla_utils import config_to_sequence


def run(args):

    # validate args
    if args.split_dir is None:
        split_path = None
    else:
        split_path = Path(args.split_dir)
    if args.data_dir is None:
        data_path = None
    else:
        data_path = Path(args.data_dir)

    # run experiment
    experiment = Experiment(
        Path(args.result_dir),
        data_path,
        args.experiment_name,
        use_processed_data=args.use_processed_data,
        verbose=args.verbose,
    )
    experiment.prepare_dataset(args.dataset_name)
    experiment.prepare_split(args.dataset_name, args.split_type, split_path=split_path)
    experiment.run_experiment(
        args.dataset_name,
        args.split_type,
        args.alg_name,
        args.num_samples,
        args.alg_seed,
        args.param_seed,
    )
    if args.write_zip:
        # write a zip of the results in folder result_dir
        experiment.zip("results.zip")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(dest="subparser_name")

    # command line parser
    cli_parser = subparsers.add_parser("cli")
    cli_parser.add_argument(
        "data_dir",
        type=str,
        help="directory containing downloaded datasets",
        required=False,
    )
    cli_parser.add_argument(
        "dataset_name",
        type=str,
        help="name of dataset. we use this to find the dataset and split.",
        required=True,
    )
    cli_parser.add_argument(
        "split_type",
        type=str,
        help="name of datasplitter to use. we use this to find the split directory.",
        required=True,
    )
    cli_parser.add_argument(
        "split_seed",
        type=int,
        default=0,
        help="random seed passed to datasplitter. only used for random splits.",
    )
    cli_parser.add_argument(
        "alg_seed",
        type=int,
        default=0,
        help="random seed passed to the recommender algorithm. only for random algorithms.",
    )
    cli_parser.add_argument(
        "param_seed",
        type=int,
        default=0,
        help="random seed for generating random hyperparameters.",
    )
    cli_parser.add_argument(
        "alg_name", type=str, help="name of the algorithm to use.", required=True,
    )
    cli_parser.add_argument(
        "num_samples",
        type=int,
        help="number of hyperparameter samples.",
        required=True,
    )
    cli_parser.add_argument(
        "result_dir",
        type=str,
        help="directory where result dir structure will be written. this directory should exist.",
        required=True,
    )
    cli_parser.add_argument(
        "experiment_name",
        type=str,
        help="name of the result directory that will be created.",
        required=True,
    )
    cli_parser.add_argument(
        "split_dir",
        type=str,
        help="if the split has been prepared, pass the directory here.",
        default=None,
    )
    cli_parser.add_argument(
        "use_processed_data",
        action="store_true",
        help="if provided, try to read the dataset at data_dir/<dataset_name>. this can be used to create splits.",
    )
    cli_parser.add_argument(
        "write_zip",
        action="store_true",
        help="if provided, zip the result directory and place it in result directory",
    )
    cli_parser.add_argument(
        "verbose", action="store_true", help="if provided, print additional output",
    )

    # config file parser
    config_parser = subparsers.add_parser("config")
    config_parser.add_argument(
        "config_file",
        type=str,
        default=None,
        help="if provided, parse args from file rather than command line. this will override any cli args provided",
        required=True,
    )

    args = parser.parse_args()

    # read args from file rather than cli if the config parser is used
    if args.subparser_name == "config":
        print(f"reading config file: {args.config_file}")
        args = parser.parse_args(["cli"] + config_to_sequence(args.config_file))

    run(args)
