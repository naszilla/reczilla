"""
run a single experiment to generate results.

here, a single "experiment" is when we train one recsys algorithm on one train/test/validation split of a dataset, with
one or more sets of hyperparameters for the algorithm. we record the train and test performance for all trained algorithms

for each experiment we regenerate the dataset split. this is reasonable because the dataset splits are O(1mb) in size,
and they are deterministic since we pass random seeds to the split function.
"""
import argparse
import os
from pathlib import Path
from Experiment_handler.Experiment import Experiment
from Utils.reczilla_utils import config_to_sequence


def run(args):

    # run experiment
    experiment = Experiment(Path(args.result_dir), args.experiment_name)
    experiment.prepare_dataset(args.data_dir, args.dataset_name)
    experiment.prepare_split(args.dataset_name, args.split_type)
    experiment.run_experiment(
        args.dataset_name,
        args.split_type,
        args.alg_name,
        args.num_samples,
        args.alg_seed,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(dest="subparser_name")

    # command line parser
    cli_parser = subparsers.add_parser("cli")
    cli_parser.add_argument(
        "--data-dir",
        type=str,
        help="directory containing downloaded datasets",
        required=True,
    )
    cli_parser.add_argument(
        "--dataset-name",
        type=str,
        help="name of dataset. this must be a subdirectory of data-dir",
        required=True,
    )
    cli_parser.add_argument(
        "--split-type", type=str, help="name of datasplitter to use.", required=True,
    )
    cli_parser.add_argument(
        "--split-seed",
        type=int,
        default=0,
        help="random seed passed to datasplitter. only used for random splits.",
    )
    cli_parser.add_argument(
        "--alg-seed",
        type=int,
        default=0,
        help="random seed passed to the recommender algorithm. only for random algorithms.",
    )
    cli_parser.add_argument(
        "--alg-name", type=str, help="name of the algorithm to use.", required=True,
    )
    cli_parser.add_argument(
        "--num-samples",
        type=int,
        help="number of hyperparameter samples.",
        required=True,
    )
    cli_parser.add_argument(
        "--result-dir",
        type=str,
        help="directory where result dir structure will be written. this directory should exist.",
        required=True,
    )
    cli_parser.add_argument(
        "--experiment-name",
        type=str,
        help="name of the result directory that will be created.",
        required=True,
    )

    # config file parser
    config_parser = subparsers.add_parser("config")
    config_parser.add_argument(
        "--config-file",
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
