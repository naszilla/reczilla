# run a random hyperparameter search for a single algorithm on a single dataset split
# assume a leave-k-out split. never download the original data

import argparse
import os

from Base.Evaluation.Evaluator import EvaluatorHoldout
from Data_manager.DataSplitter_leave_k_out import DataSplitter_leave_k_out
from ParameterTuning.RandomSearch import RandomSearch
from algorithm_handler import algorithm_handler
from dataset_handler import dataset_handler

CUTOFF_LIST = [1, 5, 10, 20, 50, 100]


def run(args):

    # check whether results exist. do this by looking for a log file produced by this function
    file_root = f"{args.datareader_name}_{args.alg_name}"
    log_file = os.path.join(args.result_dir, file_root + ".log")
    metadata_file = os.path.join(
        args.result_dir, file_root + "_metadata.zip"
    )  # this will be created by the parameter search

    # if the log file exists, do nothing
    if os.path.exists(log_file):
        print(
            f"log file found for dataset={args.datareader_name}, alg={args.alg_name} in dir={args.result_dir}. doing nothing"
        )
        return

    # initialize data reader
    data_reader = dataset_handler(args.datareader_name)(
        reload_from_original_data="as-needed", verbose=True
    )
    if args.original_data_dir != "":
        # we only need to laod the data to create a new split
        _ = data_reader.load_data(save_folder_path=args.original_data_dir)

    # initialize data splitter & load
    dataSplitter = DataSplitter_leave_k_out(
        data_reader, forbid_new_split=args.forbid_new_split
    )
    dataSplitter.load_data(save_folder_path=args.data_split_dir)
    URM_train, URM_validation, URM_test = dataSplitter.get_holdout_split()

    # create evaluators
    evaluator_validation = EvaluatorHoldout(
        URM_validation, cutoff_list=CUTOFF_LIST, exclude_seen=False
    )
    evaluator_test = EvaluatorHoldout(
        URM_test, cutoff_list=CUTOFF_LIST, exclude_seen=False
    )

    # get a recommender class, hyperparameter search space, and search_input_recommender_args from the algorithm handler
    (
        alg,
        parameter_search_space,
        search_input_recommender_args,
        max_points,
    ) = algorithm_handler(args.alg_name)

    # add the training dataset to recommender_input_args (this is then passed to the alg constructor...)
    search_input_recommender_args.CONSTRUCTOR_POSITIONAL_ARGS = [URM_train]

    # create a search object for the random parameter search
    # we need to re-initialize this for each algorithm
    parameterSearch = RandomSearch(
        alg, evaluator_validation=evaluator_validation, evaluator_test=evaluator_test,
    )

    # run a random parameter search
    parameterSearch.search(
        search_input_recommender_args,
        parameter_search_space,
        n_samples=args.num_samples,
        output_folder_path=args.result_dir,
        output_file_name_root=file_root,
        sampler_type="Sobol",
        sampler_args={},
        sample_seed=args.param_seed,
    )

    # make sure the metadata file exists
    if not os.path.exists(metadata_file):
        raise Exception(f"metadata file not found. expected: {metadata_file}")

    # write the log file
    with open(log_file, "w") as f:
        f.write(f"args: {args}\n")
        f.write(f"metadata: {metadata_file}")

    print("run_random_search done")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datareader-name", type=str, help="name of the dataset reader", required=True
    )
    parser.add_argument("--alg-name", type=str, help="name of algorithm", required=True)
    parser.add_argument(
        "--num-samples",
        type=int,
        help="number of random parameter samples",
        required=True,
    )
    parser.add_argument(
        "--param-seed",
        type=int,
        help="random seed for parameter selection",
        required=True,
    )
    parser.add_argument(
        "--original-data-dir",
        type=str,
        default="",
        help="directory of the original dataset",
        required=False,
    )
    parser.add_argument(
        "--data-split-dir", type=str, help="directory of the data split", required=True
    )
    parser.add_argument(
        "--forbid-new-split",
        help="if set, do not attempt to create a new dataset split if the split is not found",
        default=False,
        action="store_true",
    )

    # output
    parser.add_argument("--result-dir", type=str, help="directory to write results")

    return parser


if __name__ == "__main__":
    parser = parse_args()
    args = parser.parse_args()
    run(args)
