from Metafeatures.utils import register_func
from collections import OrderedDict

feature_func_lookup = {}

feature_list = [
    ("num_users",
     OrderedDict()),
    ("num_items",
     OrderedDict()),
    ("num_interactions",
     OrderedDict()),
    ("sparsity",
     OrderedDict()),
    ("item_user_ratio",
     OrderedDict())
]

# Number of users
@register_func(feature_func_lookup)
def num_users(train_set):
    """
    Number of users
    Args:
        train_set: train set as URM.

    Returns:
        Dictionary with the result
    """
    return {"": train_set.shape[0]}

# Number of items
@register_func(feature_func_lookup)
def num_items(train_set):
    """
    Number of items
    Args:
        train_set: train set as URM.

    Returns:
        Dictionary with the result
    """
    return {"": train_set.shape[1]}


@register_func(feature_func_lookup)
def num_interactions(train_set):
    """
    Number of interactions
    Args:
        train_set: train set as URM.

    Returns:
        Dictionary with the result
    """
    return {"": train_set.nnz}

@register_func(feature_func_lookup)
def sparsity(train_set):
    """
    Sparsity
    Args:
        train_set: train set as URM.

    Returns:
        Dictionary with the result
    """
    return {"": 1 - train_set.nnz / (train_set.shape[0]*train_set.shape[1])}

@register_func(feature_func_lookup)
def item_user_ratio(train_set):
    """
    Item to user ratio
    Args:
        train_set: train set as URM.

    Returns:
        Dictionary with the result
    """
    return {"": train_set.shape[1] / train_set.shape[0]}