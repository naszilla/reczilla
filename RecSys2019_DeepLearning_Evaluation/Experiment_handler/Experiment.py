"""
base class for running experiments

here, a single "experiment" is when we train one recsys algorithm on one train/test/validation split of a dataset, with
one or more sets of hyperparameters for the algorithm. we record the train and test performance for all trained algorithms
"""
import multiprocessing
import os
from pathlib import Path
from typing import List
import string
import random
import zipfile
import shutil

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

TIME_FORMAT = "%Y%m%d_%H%M%S"


class Result(object):
    """base class for experiment results"""

    def __init__(
        self,
        name: str,
        datetime_str: str,
        dataset_name: str,
        alg_name: str,
        split_name: str,
        alg_seed: int,
        param_seed: int,
        result_file: Path,
    ):
        self.name = name
        self.datetime_str = datetime_str
        self.dataset_name = dataset_name
        self.alg_name = alg_name
        self.split_name = split_name
        self.alg_seed = alg_seed
        self.param_seed = param_seed
        self.result_file = (
            result_file  # Path to the result file produced by Experiment.run_experiment
        )

    @classmethod
    def from_zip(cls, zip_path: Path, new_base_path: Path):
        """
        args:
        - zip_path: path to zip file to extract, produced by Experiment.zip()
        - new_base_path: all results will be moved to this base directory,
            to the subdir new_base_path/<dataset>/<split>/<alg>/

        initialize a result from a zip archive produced by Experiment.zip()

        first unzip into a temporary directory (in the same dir that contains the zip),
        then create the result, move it to the new base dir
        then delete the temporary dir
        """

        # create temp dir
        temp_dir = zip_path.parent.joinpath(
            "TEMP_"
            + time_to_str(TIME_FORMAT)
            + "_"
            + "".join(random.choices(string.ascii_uppercase + string.digits, k=3))
        )

        # extract the zip to the temp dir
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(temp_dir)

        # find all results, i.e. files that end in '_metadata.zip'
        result_files = [f for f in temp_dir.rglob("*_metadata.zip")]

        assert (
            len(result_files) == 1
        ), f"multiple results found in zip archive: {zip_path}. we expect only one."
        result_file = result_files[0]

        # gather the name of the alg, the split, and the dataset from the result path
        alg_name = result_file.parent.name
        split_name = result_file.parent.parent.name
        dataset_name = result_file.parent.parent.parent.name
        experiment_name = result_file.parent.parent.parent.parent.name

        # in the new base: make the result directory if it doesn't exist, and move the result zip there
        new_home = new_base_path.joinpath(dataset_name, split_name, alg_name)
        new_home.mkdir(parents=True, exist_ok=True)
        result_file = result_file.rename(new_home.joinpath(result_file.name))

        # read the alg seed, param seed, and timestamp from the zip file
        filename_split = result_file.name.split("_")
        assert filename_split[0][:7] == "algseed"
        alg_seed = int(filename_split[0][7:])
        assert filename_split[1][:9] == "paramseed"
        param_seed = int(filename_split[1][9:])

        time_str = filename_split[2] + "_" + filename_split[3]

        # finally, remove the temp directory
        shutil.rmtree(str(temp_dir))

        # return a new result object
        return cls(
            experiment_name,
            time_str,
            dataset_name,
            alg_name,
            split_name,
            alg_seed,
            param_seed,
            result_file,
        )


class Experiment(object):
    """
    class for generating and managing reczilla experiment results.

    the default constructor takes a single argument (base_directory): the name of a directory where results will be
     written. this directory should not exist, but its parent directory must exist.

    the Experiment_handler object will create this directory and write results to it.

    results are written and read according to the following convention:
    base_directory/<dataset>/<split>/<algorithm>/<result>_metadata.zip

    where <result> includes a timestamp (for now, only a timestamp).
    """

    def __init__(
        self,
        base_directory: Path,
        name: str,
        use_processed_data: bool = False,
        data_directory: Path = None,
        verbose: bool = True,
        log_file: str = None,
    ):
        """
        args:
        - base_directory: an existing directory where the experiment directory structure will be written
        - experiment_name: the name of the directory where results will be written. if it doesn't exist, create it.
        - data_directory: (optional) directory of original processed data. only used if use_processed_data=True.
        - use_processed_data: if True, attempt to read data from the data_directory. otherwise, just read dataset splits
            from paths passed as args.
        """
        self.logger = get_logger(logfile=log_file)

        # define the result & data directory
        self.base_directory = base_directory.resolve()
        self.result_directory = self.base_directory.joinpath(name)
        self.name = name

        self.verbose = verbose
        self.use_processed_data = use_processed_data  # if true, try to read the original dataset, which can be used to create splits. if false, all splits must already exist.

        # make sure the base directory exists
        assert (
            self.base_directory.exists()
        ), f"base_directory does not exist: {str(self.base_directory)}"

        if data_directory is not None:
            assert (
                not self.use_processed_data
            ), f"data_directory must be provided if use_processed_data = True "
            self.data_directory = data_directory.resolve()

            assert (
                self.data_directory.exists()
            ), f"data_directory does not exist: {str(self.data_directory)}"
        else:
            self.data_directory = None

        self.logger.info(
            f"initializing Experiment: base_directory={self.base_directory}, result_directory={self.result_directory}, data_directory={self.data_directory}"
        )

        # if this directory doesn't exist, create it
        if not self.result_directory.exists():
            self.result_directory.mkdir()
            self.logger.info(f"created result directory: {self.result_directory}")
        else:
            self.logger.info(f"found result directory: {self.result_directory}")

        self.prepared_split_dict = {}  # keys = dataset names, values = split names
        self.dataset_dict = (
            {}
        )  # keys = dataset names, values = reader objects (if use_processed_data=True) or None (if use_processed_data=False)

    def get_dataset_result_path(self, dataset_name: str):
        """get path of results for a particluar dataset"""
        return self.result_directory.joinpath(dataset_name)

    def get_split_result_path(self, dataset_name: str, split_name: str):
        """get path of results for a particluar dataset and split"""
        return self.result_directory.joinpath(dataset_name, split_name)

    def get_alg_result_path(self, dataset_name: str, split_name: str, alg_name: str):
        """get path of results for a particluar dataset and split and algorithm"""
        return self.result_directory.joinpath(dataset_name, split_name, alg_name)

    def get_dataset_path(self, dataset_name: str):
        """get path of a dataset"""
        return self.data_directory.joinpath(dataset_name)

    def get_split_path(self, dataset_name: str, split_name: str):
        """get path of dataset split"""
        return self.data_directory.joinpath(dataset_name, split_name)

    def zip(self, filename: Path):
        """zip the result directory to the file at the given path"""
        make_archive(
            str(self.result_directory), str(self.base_directory.joinpath(filename))
        )
        self.logger.info(
            f"zipped experiment directory to {str(self.base_directory)}/{filename}"
        )

    def prepare_dataset(self, dataset_name: str):
        """
        keep track of the dataset and reader object.
        if self.use_processed_data, make sure that we can read the dataset.
        """
        if dataset_name in self.dataset_dict:
            self.logger.info(f"dataset already prepared: {dataset_name}")

        # if self.use_processed_data = True, then try to read the dataset. Otherwise just make sure the directory exists.
        if self.use_processed_data:
            # -- make sure dataset exists --
            assert self.get_dataset_path(
                dataset_name
            ).exists(), f"dataset directory not found: {str(self.get_dataset_path(dataset_name))}"
            # never reload the original dataset (reload_from_original_data="never")
            self.dataset_dict[dataset_name] = dataset_handler(dataset_name)(
                reload_from_original_data="never",
                folder=str(self.get_dataset_path(dataset_name)),
            )

            # make sure the data exists in self.data_directory/dataset_name
            _ = self.dataset_dict[dataset_name].load_data()
        else:
            self.dataset_dict[dataset_name] = None

        # initialize split dict for this dataset
        self.prepared_split_dict[dataset_name] = {}
        self.logger.info(f"initialized dataset in {dataset_name}")

    def prepare_split(
        self,
        dataset_name,
        split_type,
        split_args: dict = None,
        split_path: Path = None,
    ):
        """
        check whether a split already exists. if it does not exist, create it if self.use_processed_data.

        if split_path is not None, read the data directly from this path.
        """
        if split_args is None:
            split_args = {}

        if (split_path is None) and (not self.use_processed_data):
            raise Exception(
                f"no split_path provided. if use_processed_data = False, then split_path must be provided."
            )

        # dataset must be initialized
        assert (
            dataset_name in self.dataset_dict
        ), f"dataset '{dataset_name}' must be  initialized with prepare_dataset()"

        # path to split data
        if split_path is not None:
            assert split_path.exists(), f"data_path does not exist: {str(split_path)}"
            split_path = split_path
        else:
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
                data_reader, folder=str(split_path), verbose=self.verbose, **init_kwargs
            )
            data_splitter.load_data()
            self.logger.info(f"found a split in directory {str(split_path)}")

        except FileNotFoundError:

            if not self.use_processed_data:
                self.logger.info(
                    f"split not found and use_processed_data=False. raising Exception"
                )
                raise FileNotFoundError
            else:
                self.logger.info(
                    f"split not found in directory {str(split_path)}. creating a new split."
                )

                if split_type not in SPLITTER_DICT:
                    raise Exception(f"split_type not recognized: {split_type}")

                data_splitter = SPLITTER_DICT[split_type](
                    self.dataset_dict[dataset_name],
                    **split_args,
                    folder=str(split_path),
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
        param_seed: int,
        original_split_path: str,
        cutoff_list: List[int] = None,
        result_dir: Path = None,
        time_limit=1e10,
    ):
        """
        run an experiment, writing the results in the appropriate metadata files

        if result_dir is provided, write the result here.
        otherwise, write it in the directory structure base/<dataset>/<split>/<alg>/

        time_limit is in seconds
        """
        assert (
            dataset_name in self.dataset_dict
        ), f"dataset {dataset_name} has not been prepared. call prepare_dataset first"
        assert (
            split_name in self.prepared_split_dict[dataset_name]
        ), f"split has not been prepared. call prepare_split first."

        if cutoff_list is None:
            cutoff_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 40, 50]

        # prepare evaluators
        # TODO: we might want to use the DataSplitter function get_holdout_split, but this has a different return value depending on class-specific params. which is annoying. so we will access the split dict directly
        urm_dict = self.prepared_split_dict[dataset_name][split_name].SPLIT_URM_DICT

        if "URM_validation" in urm_dict:
            self.logger.info(
                f"WARNING: URM_validation not found in URM_dict for split {dataset_name}/{split_name}"
            )
            evaluator_validation = EvaluatorHoldout(
                urm_dict["URM_validation"], cutoff_list, exclude_seen=False
            )
        else:
            evaluator_validation = None

        evaluator_test = EvaluatorHoldout(
            urm_dict["URM_test"], cutoff_list, exclude_seen=False
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
            verbose=self.verbose,
            logger=self.logger,
        )

        # pass these to RandomSearch.search(), which will add this to the metadata dict
        search_param_dict = {
            "time": time_to_str(TIME_FORMAT),
            "dataset_name": dataset_name,
            "split_name": split_name,
            "alg_name": alg_name,
            "num_samples": num_samples,
            "alg_seed": alg_seed,
            "param_seed": param_seed,
            "cutoff_list": cutoff_list,
            "experiment_name": self.name,
            "original_split_path": original_split_path,
        }

        if result_dir is not None:
            experiment_result_dir = result_dir
            assert (
                experiment_result_dir.exists()
            ), f"result_dir does not exist: {result_dir}"
        else:
            experiment_result_dir = self.get_alg_result_path(
                dataset_name, split_name, alg_name
            )

        self.logger.info(
            f"starting experiment, writing results to {str(experiment_result_dir)}"
        )

        # run a random parameter search
        time_str = time_to_str(TIME_FORMAT)
        output_file_name = f"result_" + time_str

        search_args = (search_input_recommender_args, parameter_search_space)
        search_kwargs = {
            "n_samples": min(num_samples, max_points),
            "output_folder_path": str(experiment_result_dir) + os.sep,
            "output_file_name_root": output_file_name,
            "sampler_type": "Sobol",
            "sampler_args": {},
            "param_seed": param_seed,
            "alg_seed": alg_seed,
            "metadata_dict": {"search_params": search_param_dict},
        }

        # start a process for running the search. use this to keep track of the time limit
        p = multiprocessing.Process(
            target=parameter_search.search, args=search_args, kwargs=search_kwargs
        )
        p.start()
        p.join(time_limit)

        if p.is_alive():
            self.logger.info("time limit reached. stopping search")
            p.terminate()

        # make sure that result (metadata) file exists, and add it to the list
        result_file = experiment_result_dir.joinpath(output_file_name + "_metadata.zip")
        assert result_file.exists()

        self.logger.info(f"results written to file: {str(result_file)}")

        return result_file
