from Data_manager.DataSplitter import DataSplitter
import Metafeatures.DistributionFeatures
import Metafeatures.Landmarkers
import Metafeatures.Basic

from pathlib import Path
import pandas as pd
import time


dataset_folder = "../all_data"

# Lookup dictionary for feature functions
feature_func_lookup = {}
feature_func_lookup.update(Metafeatures.Basic.feature_func_lookup)
feature_func_lookup.update(Metafeatures.DistributionFeatures.feature_func_lookup)
feature_func_lookup.update(Metafeatures.Landmarkers.feature_func_lookup)

# Feature function list.
# Entries are (function_name, function_kwargs)
all_features = (Metafeatures.Basic.feature_list +
                Metafeatures.DistributionFeatures.feature_list +
                Metafeatures.Landmarkers.feature_list)

def feature_string(func_name, func_kwargs):
    """
    Get string representation from a feature setting.
    Args:
        func_name: Name of the function to be called to extract the feature.
        func_kwargs: Keyword arguments for the function. Use OrderedDict() so that the string representation is
            consistent.

    Returns:
        String representation of the feature. If multiple features are extracted by the function, this will be the
        prefix used for the feature names.
    """
    if not func_kwargs:
        return "f_" + func_name
    return "f_" + "__".join([func_name] + ["{}_{}".format(key, val) for key, val in func_kwargs.items()])


def featurize_train_set(train_set, feature_func_list=None):
    """
    Featurize a train set using the features specified in feature_func_list.
    Args:
        train_set: URM of train set.
        feature_func_list: list containing tuples of (func_name, func_kwargs), where func_name is a string representing
        the name of the function to be called, and func_kwargs are its keyword arguments.

    Returns:
        feature_vals: Dictionary with the features, plu
    """
    if not feature_func_list:
        feature_func_list = all_features

    feature_vals = {}
    for func_name, func_kwargs in feature_func_list:
        feature_name = feature_string(func_name, func_kwargs)
        new_values = feature_func_lookup[func_name](train_set, **func_kwargs)
        if len(new_values) == 1 and list(new_values.keys())[0] == "":
            feature_vals[feature_name] = new_values[""]
        else:
            feature_vals.update({"__".join([feature_name, suffix]): value for suffix, value in new_values.items()})
    return feature_vals

def featurize_dataset_split(dataset_split_path, feature_func_list=None, feature_str_list=None):
    """
    Featurize the dataset split specified in the path. Return features as dictionary.
    Args:
        dataset_split_path: Folder that contains the dataset split. Must be a Path object.
        feature_func_list: List of feature functions to call on the dataset, specified as string, kwarg pairs.
        feature_str_list: Names of features to be extracted (alternative to specifying feature_func_list). Only these
            will be kept in the output. Cannot be specified at the same time as feature_func_list. This is useful when
            a feature function returns multiple features, and only a few of those are of interest.

    Returns:
        feature_vals: Dictionary from feature names to values. Two extra entries are added: dataset_name and split_name
    """
    if feature_str_list is not None:
        if feature_func_list is not None:
            raise RuntimeError("Cannot specify both feature_list and feature_str_list.")
        feature_func_list = [entry for entry in all_features if any(feat_str_name.startswith(feature_string(*entry))
                                                                    for feat_str_name in feature_str_list)]

    dataReader_object, splitter_class, init_kwargs = DataSplitter.load_data_reader_splitter_class(dataset_split_path)
    init_kwargs["forbid_new_split"] = True
    splitter = splitter_class(
        dataReader_object, folder=str(dataset_split_path.resolve()), verbose=True,
        **init_kwargs
    )
    splitter.load_data()

    train_set = splitter.SPLIT_URM_DICT["URM_train"]

    feature_vals = featurize_train_set(train_set, feature_func_list=feature_func_list)
    if feature_str_list is not None:
        feature_vals = {key: feature_vals[key] for key in feature_str_list}

    feature_vals.update({
        "dataset_name": dataReader_object.__class__.__name__,
        "split_name": splitter_class.__name__,
    })
    return feature_vals


def featurize_all_datasets(folder=dataset_folder):
    """
    Featurize all of the dataset splits in the specified folder. Save all to metafeatures.csv.
    Args:
        folder: String representing the path of the folder.

    Returns:

    """
    extra_info_cols = ["bucket_path", "dataset_name", "split_name"]

    # Ensure that only features that have not been previously extracted are computed
    output_file = Path("Metafeatures.csv")
    if output_file.exists():
        pre_computed_features = pd.read_csv(output_file)
        omit_features = [col for col in pre_computed_features.columns if col not in extra_info_cols]
        feature_list = [(func_name, func_kwargs) for (func_name, func_kwargs) in all_features
                        if not any(omit_feat.startswith(feature_string(func_name, func_kwargs))
                                   for omit_feat in omit_features)]
        if not feature_list:
            print("All features have been precomputed. Aborting.")
            return
        print("Found {} new features to compute.".format(len(feature_list)))
    else:
        print("No metafeatures file found. Generating from scratch.")
        pre_computed_features = None
        feature_list = None

    # Walk the folder and find all datasets. Extract features.
    path = Path(folder)
    error_paths = []
    dataset_features = []
    for path_match in path.glob("**/data_reader_splitter_class"):
        gcloud_path = path_match.parent.relative_to(folder)
        print(path_match)
        try:
            dataset_features.append(featurize_dataset_split(path_match.parent, feature_func_list=feature_list))
            dataset_features[-1]["bucket_path"] = gcloud_path.as_posix()
        except Exception as e:
            print(path_match.parent)
            error_paths.append(path_match.parent)
            raise e

    dataset_features = pd.DataFrame(dataset_features)
    if pre_computed_features is not None:
        dataset_features = pd.merge(dataset_features, pre_computed_features, how='outer', on=extra_info_cols)
    dataset_features = dataset_features[extra_info_cols + sorted([col for col in dataset_features
                                                                  if col not in extra_info_cols])]
    dataset_features.to_csv(output_file, index=False)

    print("{} datasets not processed.".format(len(error_paths)))
    print([entry.parent.name for entry in error_paths])

if __name__ == '__main__':
    start = time.time()
    featurize_all_datasets()
    print("Elapsed time: {}".format(time.time()-start))