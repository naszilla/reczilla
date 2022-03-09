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

    # False = we don't read from the processed datasets. Instead we read the split data directly
    # for this reason we don't need data_path, which points to the processed data.
    use_processed_data = False
    data_path = None

    verbose = False

    # run experiment
    experiment = Experiment(
        Path(args.result_dir),
        data_path,
        args.experiment_name,
        use_processed_data=use_processed_data,
        verbose=verbose,
    )
    experiment.prepare_dataset(args.dataset_name)
    experiment.prepare_split(args.dataset_name, args.split_type, split_path=Path(args.split_dir))
    experiment.run_experiment(
        args.dataset_name,
        args.split_type,
        args.alg_name,
        args.num_samples,
        args.alg_seed,
        args.param_seed,
    )

    # write a zip of the results in folder result_dir
    experiment.zip("results.zip")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(dest="subparser_name")

    # command line parser
    cli_parser = subparsers.add_parser("cli")
    cli_parser.add_argument(
        "dataset_name",
        type=str,
        help="name of dataset. we use this to find the dataset and split.",
    )
    cli_parser.add_argument(
        "split_type",
        type=str,
        help="name of datasplitter to use. we use this to find the split directory.",
    )
    cli_parser.add_argument(
        "alg_name", type=str, help="name of the algorithm to use.",
    )
    cli_parser.add_argument(
        "split_dir",
        type=str,
        help="directory containing split data files.",
    )
    cli_parser.add_argument(
        "alg_seed",
        type=int,
        help="random seed passed to the recommender algorithm. only for random algorithms.",
    )
    cli_parser.add_argument(
        "param_seed",
        type=int,
        help="random seed for generating random hyperparameters.",
    )
    cli_parser.add_argument(
        "num_samples",
        type=int,
        help="number of hyperparameter samples.",
    )
    cli_parser.add_argument(
        "result_dir",
        type=str,
        help="directory where result dir structure will be written. this directory should exist.",
    )
    cli_parser.add_argument(
        "experiment_name",
        type=str,
        help="name of the result directory that will be created.",
    )

    # TBD: remove config parser since we're no longer using config files
    # # config file parser
    # config_parser = subparsers.add_parser("config")
    # config_parser.add_argument(
    #     "config_file",
    #     type=str,
    #     default=None,
    #     help="if provided, parse args from file rather than command line. this will override any cli args provided",
    #     required=True,
    # )
    #
    args = parser.parse_args()
    #
    # # read args from file rather than cli if the config parser is used
    # if args.subparser_name == "config":
    #     print(f"reading config file: {args.config_file}")
    #     args = parser.parse_args(["cli"] + config_to_sequence(args.config_file))

    run(args)
