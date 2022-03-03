"""
base classes for experiment results

here, a single "experiment" is when we train one recsys algorithm on one train/test/validation split of a dataset, with
one or more sets of hyperparameters for the algorithm. we record the train and test performance for all trained algorithms
"""
import os
from pathlib import Path
from typing import List
from Utils.reczilla_utils import make_archive

from Base.Evaluation.Evaluator import EvaluatorHoldout
from Data_manager.DataSplitter import DataSplitter
from Data_manager.DataSplitter_leave_k_out import DataSplitter_leave_k_out
from Data_manager.DataSplitter_k_fold_random import DataSplitter_k_fold_random
from ParameterTuning.RandomSearch import RandomSearch
from Utils.reczilla_utils import get_logger, time_to_str
from algorithm_handler import algorithm_handler
from dataset_handler import dataset_handler

SPLITTER_DICT = {
    "DataSplitter_leave_k_out": DataSplitter_leave_k_out,
    "DataSplitter_k_fold_random": DataSplitter_k_fold_random,
}


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
        self.base_directory = base_directory.resolve()
        # make sure the base directory exists
        assert (
            base_directory.exists()
        ), f"base_directory does not exist: {str(base_directory)}"

        # define the result directory
        self.result_directory = base_directory.joinpath(experiment_name).resolve()

        print(f"EXPERIMENT: base_directory: {self.base_directory}")
        print(f"EXPERIMENT: result_directory: {self.result_directory}")

        # if this directory doesn't exist, create it
        if not self.result_directory.exists():
            self.result_directory.mkdir()
            self.logger.info(f"created result directory: {self.result_directory}")
        else:
            self.logger.info(f"found result directory: {self.result_directory}")

        self.prepared_split_dict = {}  # keys = dataset names, values = split names
        self.dataset_dict = {}  # keys = dataset names, values = reader objects
        self.result_list = []

    def get_dataset_path(self, dataset_name):
        """get path of results for a particluar dataset"""
        return self.result_directory.joinpath(dataset_name)

    def get_split_path(self, dataset_name, split_name):
        """get path of results for a particluar dataset and split"""
        return self.result_directory.joinpath(dataset_name, split_name)

    def get_alg_path(self, dataset_name, split_name, alg_name):
        """get path of results for a particluar dataset and split and algorithm"""
        return self.result_directory.joinpath(dataset_name, split_name, alg_name)

    def zip(self, filename):
        """zip the result directory to the file at the given path"""
        make_archive(
            str(self.result_directory), str(self.base_directory.joinpath(filename))
        )
        self.logger.info(
            f"zipped experiment directory to {str(self.base_directory)}/{filename}"
        )

    def prepare_dataset(self, data_dir, dataset_name):
        """keep track of the dataset and reader object"""
        if dataset_name in self.dataset_dict:
            self.logger.info(f"dataset already prepared: {dataset_name}")

        # -- make sure dataset exists --
        # never reload the original dataset (reload_from_original_data="never")
        self.dataset_dict[dataset_name] = dataset_handler(dataset_name)(
            reload_from_original_data="never",
            folder=str(Path(data_dir).joinpath(dataset_name)),
        )

        # make sure the data exists in data_dir/dataset_name
        _ = self.dataset_dict[dataset_name].load_data()

        # initialize split dict for this dataset
        self.prepared_split_dict[dataset_name] = {}
        self.logger.info(
            f"initialized dataset in {str(self.get_dataset_path(dataset_name))}"
        )

    # TODO: add random seed to splitter, and keep track of this seed (maybe in the name of the split?)
    # TODO: keep track of split params somehow...
    def prepare_split(self, dataset_name, split_type, split_args: dict = None):
        """
        check whether a split already exists. if it does not exist, create it.
        """
        if split_args is None:
            split_args = {}

        # dataset must be initialized
        assert (
            dataset_name in self.dataset_dict
        ), f"dataset '{dataset_name}' must be  initialized with prepare_dataset()"

        split_path = self.get_split_path(dataset_name, split_type)

        # if the split directory has already been prepared, skip it
        if str(split_path) in self.prepared_split_dict[dataset_name]:
            self.logger.info(f"split has already been prepared: {str(split_path)}")

        # first, attempt to read the split. if it does not exist, then create it
        try:

            (
                data_reader,
                splitter_class,
                init_kwargs,
            ) = DataSplitter.load_data_reader_splitter_class(split_path)
            data_splitter = splitter_class(
                data_reader, folder=str(split_path), **init_kwargs
            )
            data_splitter.load_data()
            self.logger.info(f"found a split in directory {str(split_path)}")

        except FileNotFoundError:

            self.logger.info(
                f"split not found in directory {str(split_path)}. creating a new split."
            )

            if split_type not in SPLITTER_DICT:
                raise Exception(f"split_type not recognized: {split_type}")

            data_splitter = SPLITTER_DICT[split_type](
                self.dataset_dict[dataset_name], **split_args, folder=str(split_path),
            )

            # write the split in the result subfolder
            data_splitter.load_data()
            self.logger.info(f"new split created.")

        assert (
            "URM_test" in data_splitter.SPLIT_URM_DICT
        ), f"URM_test not found in split: {dataset_name}/{split_type}"
        assert (
            "URM_train" in data_splitter.SPLIT_URM_DICT
        ), f"URM_train not found in split: {dataset_name}/{split_type}"

        self.prepared_split_dict[dataset_name][split_type] = data_splitter
        self.logger.info(f"initialized split {dataset_name}/{split_type}")

    def run_experiment(
        self,
        dataset_name: str,
        split_name: str,
        alg_name: str,
        num_samples: int,
        alg_seed: int,
        cutoff_list: List[int] = None,
    ):
        """
        run an experiment, writing the results in the appropriate metadata files
        """
        assert (
            dataset_name in self.dataset_dict
        ), f"dataset {dataset_name} has not been prepared. call prepare_dataset first"
        assert (
            split_name in self.prepared_split_dict[dataset_name]
        ), f"split has not been prepared. call prepare_split first."

        if cutoff_list is None:
            cutoff_list = [1, 5, 10, 50]

        # prepare evaluators
        # TODO: we might want to use the DataSplitter function get_holdout_split, but this has a different return value depending on class-specific params. which is annoying. so we will access the split dict directly
        urm_dict = self.prepared_split_dict[dataset_name][split_name].SPLIT_URM_DICT

        if "URM_validation" in urm_dict:
            self.logger.info(
                f"WARNING: URM_validation not found in URM_dict for split {dataset_name}/{split_name}"
            )
            evaluator_validation = EvaluatorHoldout(
                urm_dict["URM_validation"], cutoff_list=cutoff_list, exclude_seen=False
            )
        else:
            evaluator_validation = None

        evaluator_test = EvaluatorHoldout(
            urm_dict["URM_test"], cutoff_list=cutoff_list, exclude_seen=False
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
            urm_dict["URM_train"]
        ]

        # create a search object for the random parameter search
        # we need to re-initialize this for each algorithm
        parameter_search = RandomSearch(
            alg,
            evaluator_validation=evaluator_validation,
            evaluator_test=evaluator_test,
        )

        experiment_result_dir = self.get_alg_path(dataset_name, split_name, alg_name)

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
