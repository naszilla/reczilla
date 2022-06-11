# Run a reczilla model on a new dataset. Generate predictions. Train a new reczilla model on the dataset.

from ReczillaClassifier.get_alg_feat_selection_data import alg_feature_selection_featurized
from ReczillaClassifier.classifier import run_metalearner
from Metafeatures.Featurizer import featurize_dataset_split
from Data_manager.DataSplitter import DataSplitter
from Base.Evaluation.Evaluator import EvaluatorHoldout
from Utils.reczilla_utils import get_parameterized_alg

import pickle
import numpy as np
import argparse
import time
from pathlib import Path

# Default values for command line parser
default_dataset_name = "metadata-v2"
default_model_name = "xgboost"
default_num_algs = 10
default_num_feats = 10
metalearner_options = ["xgboost", "knn", "linear", "svm-poly"]
default_metric_name = 'PRECISION_cut_10'

def is_time_metric(metric_name):
    """
    Returns true iff the metric is to be minimized and not maximized
    Args:
        metric_name: The name of the metric.

    Returns:

    """
    return "time" in metric_name

def reczilla_train(metric_name, dataset_name=default_dataset_name, num_algs=default_num_algs,
                   num_feats=default_num_feats, model_name=default_model_name, out_filename=None):
    """
    Train a Reczilla model.
    Args:
        metric_name:
        dataset_name:
        num_algs:
        num_feats:
        model_name:
        out_filename:

    Returns:

    """

    if is_time_metric(metric_name):
        minimize_metric = True
        metric_df_name = metric_name
    else:
        metric_df_name = "test_metric_" + metric_name
        minimize_metric = False

    X_train, y_train, _, _, _, extra_outputs = \
            alg_feature_selection_featurized(metric_name=metric_df_name,
                                             test_datasets=[],
                                             dataset_name=dataset_name,
                                             num_algs=num_algs,
                                             num_feats=num_feats,
                                             get_extra_outputs=True,
                                             minimize_metric=minimize_metric
                                             )
    selected_feats, selected_algs = extra_outputs["selected_feats"], extra_outputs["selected_algs"]

    _, model = run_metalearner(model_name, X_train, y_train, X_test=None, return_model=True)
    save_dict = {"model": model,
                 "selected_feats": selected_feats,
                 "selected_algs": selected_algs,
                 "metric_name": metric_name}

    if out_filename is not None:
        with open(out_filename, "wb") as f:
            pickle.dump(save_dict, f)

    return save_dict

def reczilla_inference(model_save_dict, dataset_split_path):
    """
    Perform inference using a Reczilla model
    Args:
        model_save_dict:
        dataset_split_path:

    Returns:

    """
    selected_feats = model_save_dict["selected_feats"]
    selected_algs = model_save_dict["selected_algs"]
    metafeatures = featurize_dataset_split(dataset_split_path, feature_str_list=model_save_dict["selected_feats"])
    feat_test = np.array([metafeatures[feat_name] for feat_name in selected_feats])[np.newaxis, :]
    preds = np.squeeze(model_save_dict["model"].predict(feat_test))
    alg_perf = [(alg_name, pred) for alg_name, pred in zip(selected_algs, preds)]
    alg_perf.sort(key=lambda entry: entry[1], reverse=True)
    return alg_perf


def parse_metric_name(metric_name):
    """
    Separate a metric name into the cut value and the metric name.
    Args:
        metric_name:

    Returns:

    """
    tokens = metric_name.split("_")
    cut_idx = tokens.index("cut")
    cut = int(tokens[cut_idx+1])
    new_tokens = tokens[:cut_idx] + tokens[cut_idx+2:]
    return cut, "_".join(new_tokens)

def train_best_model(predictions, dataset_split_path, metric_name, rec_model_save_path=None):
    """
    Use the predictions output by reczilla_inference to train the best algorithm on the dataset in dataset_split_path
    Args:
        predictions:

    Returns:

    """
    if isinstance(dataset_split_path, str):
        dataset_split_path = Path(dataset_split_path)
    # Load dataset
    dataReader_object, splitter_class, init_kwargs = DataSplitter.load_data_reader_splitter_class(dataset_split_path)
    init_kwargs["forbid_new_split"] = True
    splitter = splitter_class(
        dataReader_object, folder=str(dataset_split_path.resolve()), verbose=True,
        **init_kwargs
    )
    splitter.load_data()

    # Initialize algorithm
    reverse_order = not is_time_metric(metric_name)
    alg_perf = sorted(predictions, key=lambda entry: entry[1], reverse=reverse_order)

    best_alg, best_alg_pred = alg_perf[0]
    print(f"Chose {best_alg} for {metric_name} with predicted value {best_alg_pred}")
    alg_class, alg_kwargs, search_input_recommender_args = get_parameterized_alg(best_alg)

    # Train
    start = time.time()
    recommender = alg_class(splitter.SPLIT_URM_DICT["URM_train"],
                            *search_input_recommender_args.CONSTRUCTOR_POSITIONAL_ARGS,
                            **search_input_recommender_args.CONSTRUCTOR_KEYWORD_ARGS)
    recommender.fit(**alg_kwargs)
    train_time = time.time() - start

    # Evaluate
    if not is_time_metric(metric_name):
        cut, metric_name_short = parse_metric_name(metric_name)
        evaluator_validation = EvaluatorHoldout(
            splitter.SPLIT_URM_DICT["URM_test"], [cut], exclude_seen=False
        )
        results_dict, _ = evaluator_validation.evaluateRecommender(recommender)
        best_alg_actual = results_dict[cut][metric_name_short]
    else:
        if metric_name == "time_on_train":
            best_alg_actual = train_time
        else:
            best_alg_actual = "?"
            print(f"Performance evaluation for {metric_name} not implemented")

    print()
    print("*" * 50)
    print("Done training recommender. Summary:")
    print(f"Metric to optimize: {metric_name}")
    print(f"Chosen algorithm: {best_alg}")
    print(f"Predicted performance: {best_alg_pred}")
    print(f"Actual performance: {best_alg_actual}")
    print("*" * 50)
    print()

    if rec_model_save_path is not None:
        recommender.save_model(rec_model_save_path)

# Load model
def load_reczilla_model(filename):
    with open(filename, "rb") as f:
        model_save_dict = pickle.load(f)
        return model_save_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Reczilla on a new dataset.")
    parser.add_argument('--train_meta', action='store_true',
                        help="Use to train a new metalearner Reczilla model (instead of loading).")
    parser.add_argument('--metamodel_filepath', required=True, help="Filepath of Reczilla model (to save or load).")

    # If performing inference
    parser.add_argument('--dataset_split_path',
                        help="Path of dataset split to perform inference on. Only required if performing inference")
    parser.add_argument('--rec_model_save_path',
                        help="Destination path for recommender model trained on dataset on dataset_split_path.")

    # Arguments for training reczilla
    parser.add_argument('--metadataset_name', default=default_dataset_name,
                        help="Name of metadataset (required if training metamodel).")
    parser.add_argument('--metamodel_name', choices=metalearner_options, default=default_model_name,
                        help="Name of metalearner to use (required if training metamodel).")
    parser.add_argument('--target_metric', default=default_metric_name,
                        help="Target metric to optimize.")
    parser.add_argument('--num_algorithms', default=default_num_algs, type=int,
                        help="Number of algorithms to use in Reczilla (required if training metamodel).")
    parser.add_argument('--num_metafeatures', default=default_num_feats, type=int,
                        help="Number of metafeatures to select for metalearner.")

    args = parser.parse_args()

    # Training
    if args.train_meta:
        model_save_dict = reczilla_train(args.target_metric,
                                         dataset_name=args.metadataset_name,
                                         num_algs=args.num_algorithms,
                                         num_feats=args.num_metafeatures,
                                         model_name=args.metamodel_name,
                                         out_filename=args.metamodel_filepath)
        print(f"Metamodel saved to {args.metamodel_filepath}")
    else:
        print(f"Loading metamodel from {args.metamodel_filepath}")
        model_save_dict = load_reczilla_model(args.metamodel_filepath)

    # Inference
    if args.dataset_split_path is not None:
        predictions = reczilla_inference(model_save_dict, args.dataset_split_path)
        train_best_model(predictions, args.dataset_split_path, model_save_dict["metric_name"],
                         rec_model_save_path=args.rec_model_save_path)