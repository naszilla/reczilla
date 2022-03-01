"""
base classes for experiment results

here, a single "experiment" is when we train one recsys algorithm on one train/test/validation split of a dataset, with
one or more sets of hyperparameters for the algorithm. we record the train and test performance for all trained algorithms
"""
import os
from pathlib import Path
from typing import List

from Base.Evaluation.Evaluator import EvaluatorHoldout
from Data_manager.datareader_light import datareader_light
from Data_manager.datasplitter_light import read_split, write_split
from ParameterTuning.RandomSearch import RandomSearch
from Utils.reczilla_utils import get_logger, time_to_str
from algorithm_handler import algorithm_handler


class Experiment(object):
    """
    class for generating and managing reczilla experiment results.

    the default constructor takes a single argument (base_directory): the name of a directory where results will be
     written. this directory should not exist, but its parent directory must exist.

    the Experiment_handler object will create this directory and write results to it.

    TODO: the secondary constructor reads results from a directory.

    results are written and read according to the following convention:
    base_directory/<dataset>/<split>/<algorithm>/<result>_metadata.zip

    where <result> includes a timestamp (for now, only a timestamp).
    """

    TIME_FORMAT = "%Y%m%d_%H%M%S"

    def __init__(self, base_directory: Path, experiment_name: str):
        """
        base_directory: an existing directory where the experiment directory structure will be written
        experiment_name: the name of the directory where results will be written. if it doesn't exist, create it.
        """
        self.logger = get_logger()

        # make sure the base directory exists
        assert (
            base_directory.exists()
        ), f"base_directory does not exist: {str(base_directory)}"

        # define the result directory
        self.result_directory = base_directory.joinpath(experiment_name)

        # if this directory doesn't exist, create it
        if not self.result_directory.exists():
            self.result_directory.mkdir()
            self.logger.info(f"created result directory: {self.result_directory}")
        else:
            self.logger.info(f"found result directory: {self.result_directory}")

        self.prepared_split_list = []
        self.result_list = []

    # TODO: update this after Sujay and Jonathan update the split methods
    # TODO: add random seed to splitter, and keep track of this seed (maybe in the name of the split?)
    def prepare_split(self, data_dir, dataset_name, split_type):
        """
        check whether a split already exists. if it does not exist, create it.
        """
        # # make sure the data and result directories exist
        # result_dir = Path(args.result_dir).resolve()
        # assert '~' not in args.result_dir, f"home directory not allowed in result-dir: {str(result_dir)}"
        # assert result_dir.exists(), f"result-dir does not exist: {str(result_dir)}"
        #
        # data_dir = Path(args.data_dir).resolve()
        # assert '~' not in args.data_dir, f"home directory not allowed in data-dir: {str(data_dir)}"
        # assert data_dir.exists(), f"data-dir does not exist: {str(data_dir)}"
        #

        split_dir = self.result_directory.joinpath(dataset_name, split_type)

        # if the split directory has already been prepared, skip it
        if split_dir in self.prepared_split_list:
            self.logger.info(f"split has already been prepared: {str(split_dir)}")

        # check whether the split exists
        if split_dir.exists():
            try:
                data_dict = read_split(split_dir)
                # make sure all URMs exist
                for x in ["URM_train", "URM_test", "URM_validation"]:
                    assert x in data_dict, f"object not found in data_dict: {x}"
                self.logger.info(f"split already exists in directory {str(split_dir)}.")
                self.prepared_split_list.append(split_dir)
                return split_dir
            except Exception as e:
                self.logger.info(
                    f"found split directory, but could not read split data: {str(split_dir)}"
                )
                self.logger.info(f"exception: {e}")

        self.logger.info("creating split...")

        dataset_dir = Path(data_dir).joinpath(dataset_name)
        assert dataset_dir.exists(), f"dataset directory not found: {str(dataset_dir)}"

        # read dataset
        dataset = datareader_light(str(dataset_dir) + os.sep)

        # generate data split and write to temporary directory
        write_split(dataset, split_type, str(split_dir))
        self.logger.info(f"created split in {str(split_dir)}")

        self.prepared_split_list.append(split_dir)
        return split_dir

    def run_experiment(
        self,
        split_dir: Path,
        alg_name: str,
        num_samples: int,
        alg_seed: int,
        cutoff_list: List[int] = None,
    ):
        """
        run an experiment, writing the results in the appropriate metadata files
        """

        assert (
            split_dir in self.prepared_split_list
        ), f"split has not been prepared. call prepare_split first."

        if cutoff_list is None:
            cutoff_list = [1, 5, 10, 50]

        # read the split data
        data_dict = read_split(split_dir)

        # prepare evaluators
        evaluator_validation = EvaluatorHoldout(
            data_dict["URM_validation"], cutoff_list=cutoff_list, exclude_seen=False
        )
        evaluator_test = EvaluatorHoldout(
            data_dict["URM_test"], cutoff_list=cutoff_list, exclude_seen=False
        )

        # get a recommender class, hyperparameter search space, and search_input_recommender_args from the algorithm handler
        (
            alg,
            parameter_search_space,
            search_input_recommender_args,
            max_points,
        ) = algorithm_handler(alg_name)

        # add the training dataset to recommender_input_args (this is then passed to the alg constructor...)
        search_input_recommender_args.CONSTRUCTOR_POSITIONAL_ARGS = [
            data_dict["URM_train"]
        ]

        # create a search object for the random parameter search
        # we need to re-initialize this for each algorithm
        parameter_search = RandomSearch(
            alg,
            evaluator_validation=evaluator_validation,
            evaluator_test=evaluator_test,
        )

        experiment_result_dir = split_dir.joinpath(alg_name)

        self.logger.info(
            f"starting experiment, writing results to {str(experiment_result_dir)}"
        )

        # run a random parameter search
        output_file_name = f"seed{alg_seed}_" + time_to_str(self.TIME_FORMAT)
        parameter_search.search(
            search_input_recommender_args,
            parameter_search_space,
            n_samples=num_samples,
            output_folder_path=str(experiment_result_dir) + os.sep,
            output_file_name_root=output_file_name,
            sampler_type="Sobol",
            sampler_args={},
            sample_seed=alg_seed,
        )

        # make sure that result (metadata) file exists, and add it to the list
        result_file = experiment_result_dir.joinpath(output_file_name + "_metadata.zip")
        self.result_list.append(result_file)

        self.logger.info(f"results written to file: {str(result_file)}")
