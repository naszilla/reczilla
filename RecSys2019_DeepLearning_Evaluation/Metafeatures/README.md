
# Guide to metafeatures module

## Overview
This module provides functionality to extract metafeatures for a  dataset split (using only the training set). The main entrypoint for the functions in this module is `Featurizer.py`. The functions in this script can be used to extract metafeatures for the full metadataset and save them in a .csv file, and also to extract metafeatures given a new dataset.

## Using the featurizer

To extract all the metafeatures for all of the datasets:
1. Run `fetch_data.sh`, which gets all dataset splits from the Google Cloud bucket and places them in a local folder, `../all_data`.
2. Run `Featurizer.py`, which extracts metafeatures for all dataset splits in `../all_data`. The resulting metafeatures are saved to `Metafeatures.csv`

To extract metafeatures for a dataset split specified in some location, use `featurize_dataset_split()` within `Featurlizer.py`.

## Implementing new features

This section is only relevant if you wish to add metafeatures to the featurizer.

`Featurizer.py` keeps track of metafeatures to extract in two data structures: `feature_func_lookup` and `all_features`. A metafeature is obtained by calling a function which returns a dictionary with one or several metafeature values. If some metafeature is obtained by calling a function named `foo` with arguments given in an OrderedDict `kwargs`, then there will be an entry `("foo", kwargs)` in `all_features` (the use of an OrderedDict is so that the feature can be converted into a string consistently). Furthermore, `feature_func_lookup` must contain an entry with the string `"foo"` mapping to function `foo`.

New features may be implemented by following this general workflow, as long as `all_features` and `feature_func_lookup` in `Featurizer.py` end up containing the necessary entries. The main sets of features currently implemented are given in `Basic.py`, `Landmarkers.py`, and `DistributionFeatures.py`, from which `all_features` and `feature_func_lookup` are currently populated.