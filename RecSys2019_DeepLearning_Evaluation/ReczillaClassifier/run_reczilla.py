# Run a reczilla model on a new dataset. Generate predictions. Train a new reczilla model on the dataset.

from ReczillaClassifier.get_alg_feat_selection_data import alg_feature_selection_featurized
from ReczillaClassifier.classifier import run_metalearner
from Metafeatures.Featurizer import featurize_dataset_split
from Data_manager.DataSplitter import DataSplitter
from Base.Evaluation.Evaluator import EvaluatorHoldout

import pickle
import numpy as np
import argparse
from pathlib import Path

# Default values for command line parser
default_dataset_name = "metadata-v2"
default_model_name = "xgboost"
default_num_algs = 10
default_num_feats = 10
metalearner_options = ["xgboost", "knn", "linear", "svm-poly"]
#default_metric_name = 'test_metric_PRECISION_cut_10'
default_metric_name = 'PRECISION_cut_10'

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
    if metric_name != "training_time":
        metric_name = "test_metric_" + metric_name
    X_train, y_train, _, _, _, extra_outputs = \
            alg_feature_selection_featurized(metric_name=metric_name,
                                             test_datasets=[],
                                             dataset_name=dataset_name,
                                             num_algs=num_algs,
                                             num_feats=num_feats,
                                             get_extra_outputs=True
                                             )
    selected_feats, selected_algs = extra_outputs["selected_feats"], extra_outputs["selected_algs"]

    _, model = run_metalearner(model_name, X_train, y_train, X_test=None, return_model=True)
    save_dict = {"model": model,
                 "selected_feats": selected_feats,
                 "selected_algs": selected_algs}

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


def alg_setting_from_name(alg_param_name):
    """Return algorithm class and keyword arguments based on its alg_param_name string descriptor"""
    # return alg_class, alg_kwargs
    return None, None

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

def train_best_model(predictions, dataset_split_path, metric_name):
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
    alg_perf =  sorted(predictions, key=lambda entry: entry[1], reverse=True)
    best_alg, best_alg_pred = alg_perf[0]
    print(f"Chose {best_alg} for {metric_name} with predicted value {best_alg_pred}")
    alg_class, alg_kwargs = alg_setting_from_name(best_alg)

    # Train
    recommender = alg_class(splitter.SPLIT_URM_DICT["URM_train"])
    recommender.fit(**alg_kwargs)

    # Evaluate
    cut, metric_name = parse_metric_name(metric_name)
    evaluator_validation = EvaluatorHoldout(
        splitter.SPLIT_URM_DICT["URM_test"], [cut], exclude_seen=False
    )
    results_dict, _ = evaluator_validation.evaluateRecommender(recommender)
    best_alg_actual = results_dict[cut][metric_name]

    print(f"Actual performance: {best_alg_actual}")


# Load model
def load_reczilla_model(filename):
    with open(filename, "rb") as f:
        model_save_dict = pickle.load(f)
        return model_save_dict


# dataset_split_path = "all_data/splits-v5/CiaoDVD/DataSplitter_leave_k_out_last"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Reczilla on a new dataset.")
    parser.add_argument('--train_meta', action='store_true',
                        help="Use to train a new metalearner Reczilla model (instead of loading).")
    parser.add_argument('--metamodel_filepath', required=True, help="Filepath of Reczilla model (to save or load).")

    # If performing inference
    parser.add_argument('--dataset_split_path',
                        help="Path of dataset split to perform inference on. Only required if performing inference")

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
        train_best_model(predictions, args.dataset_split_path, args.target_metric)