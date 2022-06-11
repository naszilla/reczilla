#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generic dataset reader that creates a dataset from a URM from CSV with format (user_id, item_id, rating, timestamp).
this class can be adapted to create a new dataset reader.

@author: Duncan McElfresh
"""

import shutil
from Data_manager.Dataset import Dataset
from Data_manager.DataReader import DataReader
from Data_manager.DataReader_utils import load_CSV_into_SparseBuilder


class GenericCSVDatasetReader(DataReader):

    DATASET_NAME = "GenericCSV"
    DATASET_URL = None
    DATASET_SUBFOLDER = f"{DATASET_NAME}/"
    AVAILABLE_ICM = []
    DATASET_SPECIFIC_MAPPER = []

    IS_IMPLICIT = False

    def _get_dataset_name_root(self):
        return self.DATASET_SUBFOLDER

    def _load_from_original_file(self):
        # Load data from original

        self._print("Loading original data")

        URM_path = "../examples/random_rating_list.csv"

        (
            URM_all,
            URM_timestamp,
            item_original_ID_to_index,
            user_original_ID_to_index,
        ) = load_CSV_into_SparseBuilder(
            URM_path, separator=",", header=True, remove_duplicates=True, timestamp=True
        )

        loaded_URM_dict = {"URM_all": URM_all, "URM_timestamp": URM_timestamp}

        loaded_dataset = Dataset(
            dataset_name=self._get_dataset_name(),
            URM_dictionary=loaded_URM_dict,
            ICM_dictionary=None,
            ICM_feature_mapper_dictionary=None,
            UCM_dictionary=None,
            UCM_feature_mapper_dictionary=None,
            user_original_ID_to_index=user_original_ID_to_index,
            item_original_ID_to_index=item_original_ID_to_index,
            is_implicit=self.IS_IMPLICIT,
        )

        self._print("loading complete")

        return loaded_dataset
