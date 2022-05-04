from Data_manager.DataSplitter import DataSplitter
import Metafeatures.DistributionFeatures
import Metafeatures.Landmarkers
from Metafeatures.utils import register_func

from pathlib import Path
import pandas as pd


dataset_folder = "../all_data"

# Get string representation from a feature setting
def feature_string(func_name, func_kwargs):
    if not func_kwargs:
        return "f_" + func_name
    return "f_" + "__".join([func_name] + ["{}_{}".format(key, val) for key, val in func_kwargs.items()])

feature_func_lookup = {} # This will serve as a lookup for feature functions
feature_func_lookup.update(Metafeatures.DistributionFeatures.feature_func_lookup)
feature_func_lookup.update(Metafeatures.Landmarkers.feature_func_lookup)

# Entries are (function_name, function_kwargs)
all_features = [
    ("num_users",
     {}),
    ("num_items",
     {}),
    ("num_interactions",
     {}),
    ("sparsity",
     {}),
    ("item_user_ratio",
     {})
]

all_features += Metafeatures.DistributionFeatures.feature_list
all_features += Metafeatures.Landmarkers.feature_list

# Number of users
@register_func(feature_func_lookup)
def num_users(train_set):
    return {"": train_set.shape[0]}

# Number of items
@register_func(feature_func_lookup)
def num_items(train_set):
    return {"": train_set.shape[1]}

# Number of interactions
@register_func(feature_func_lookup)
def num_interactions(train_set):
    return {"": train_set.nnz}

# Sparsity of interaction matrix
@register_func(feature_func_lookup)
def sparsity(train_set):
    return {"": 1 - train_set.nnz / (train_set.shape[0]*train_set.shape[1])}

# Item to user ratio
@register_func(feature_func_lookup)
def item_user_ratio(train_set):
    return {"": train_set.shape[1] / train_set.shape[0]}


def featurize_train_set(train_set, feature_list=None):
    if not feature_list:
        feature_list = all_features

    feature_vals = {}
    for func_name, func_kwargs in feature_list:
        feature_name = feature_string(func_name, func_kwargs)
        #new_values = globals()[func_name](train_set, **func_kwargs)
        new_values = feature_func_lookup[func_name](train_set, **func_kwargs)
        if len(new_values) == 1 and list(new_values.keys())[0] == "":
            feature_vals[feature_name] = new_values[""]
        else:
            feature_vals.update({"__".join([feature_name, suffix]): value for suffix, value in new_values.items()})
        # feature_vals[feature_name] = globals()[func_name](train_set, **func_kwargs)
    return feature_vals

def featurize_dataset_split(dataset_split_path, feature_list=None):
    dataReader_object, splitter_class, init_kwargs = DataSplitter.load_data_reader_splitter_class(dataset_split_path)
    init_kwargs["forbid_new_split"] = True
    splitter = splitter_class(
        dataReader_object, folder=str(dataset_split_path.resolve()), verbose=True,
        **init_kwargs
    )
    splitter.load_data()

    train_set = splitter.SPLIT_URM_DICT["URM_train"]
    feature_vals = {
        "dataset_name": dataReader_object.__class__.__name__,
        "split_name": splitter_class.__name__,
    }
    feature_vals.update(featurize_train_set(train_set, feature_list=feature_list))
    return feature_vals


def featurize_all_datasets(folder=dataset_folder, omit_paths=None):
    if omit_paths is None:
        omit_paths = ["splits-v1", "splits-v2", "splits-v4"]

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
        if gcloud_path.parts[0] in omit_paths:
            continue
        print(path_match)
        #try:
        dataset_features.append(featurize_dataset_split(path_match.parent, feature_list=feature_list))
        dataset_features[-1]["bucket_path"] = gcloud_path.as_posix()
        # except Exception as e:
        #     print(e)
        #     error_paths.append(path_match.parent)
        #     import pdb
        #     pdb.set_trace()

    dataset_features = pd.DataFrame(dataset_features)
    if pre_computed_features is not None:
        dataset_features = pd.merge(dataset_features, pre_computed_features, how='outer', on=extra_info_cols)
    dataset_features = dataset_features[extra_info_cols + sorted([col for col in dataset_features if col not in extra_info_cols])]
    dataset_features.to_csv(output_file, index=False)

    print("{} datasets not processed.".format(len(error_paths)))
    print([entry.parent.name for entry in error_paths])

if __name__ == '__main__':
    featurize_all_datasets()