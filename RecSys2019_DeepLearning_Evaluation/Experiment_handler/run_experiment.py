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

from Data_manager.datareader_light import datareader_light
from Data_manager.datasplitter_light import write_split
from Experiment_handler.Experiment import Experiment


def run(args):

    # run experiment
    experiment = Experiment(Path(args.result_dir), args.experiment_name)
    split_dir = experiment.prepare_split(
        args.data_dir, args.dataset_name, args.split_type
    )
    experiment.run_experiment(split_dir, args.alg_name, args.num_samples, args.alg_seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=str,
        help="directory containing downloaded datasets",
        required=True,
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        help="name of dataset. this must be a subdirectory of data-dir",
        required=True,
    )
    parser.add_argument(
        "--split-type", type=str, help="name of datasplitter to use.", required=True,
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=0,
        help="random seed passed to datasplitter. only used for random splits.",
    )
    parser.add_argument(
        "--alg-seed",
        type=int,
        default=0,
        help="random seed passed to the recommender algorithm. only for random algorithms.",
    )
    parser.add_argument(
        "--alg-name", type=str, help="name of the algorithm to use.", required=True,
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        help="number of hyperparameter samples.",
        required=True,
    )
    parser.add_argument(
        "--result-dir",
        type=str,
        help="directory where result dir structure will be written. this directory should exist.",
        required=True,
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        help="name of the result directory that will be created.",
        required=True,
    )

    args = parser.parse_args()
    run(args)
