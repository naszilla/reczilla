#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# TODO: fetch dataset from online source

@author: Sujay Khandagale
"""

import zipfile, shutil
from io import StringIO
import pandas as pd
from Data_manager.Dataset import Dataset
from Data_manager.DataReader import DataReader
from Data_manager.DataReader_utils import download_from_URL, load_CSV_into_SparseBuilder


class RecipesReader(DataReader):

    DATASET_URL = "Data_manager/Recipes/RAW_interactions.csv"
    DATASET_SUBFOLDER = "Recipes/"
    AVAILABLE_ICM = []
    DATASET_SPECIFIC_MAPPER = []

    IS_IMPLICIT = False


    def _get_dataset_name_root(self):
        return self.DATASET_SUBFOLDER



    def _load_from_original_file(self):
        # Load data from original

        self._print("Loading original data")

        df = pd.read_csv(self.DATASET_URL)
        df = df[['user_id', 'recipe_id', 'rating', 'date']]
        df['date'] = pd.to_datetime(df['date']).apply(lambda x: x.value)

        URM_path = StringIO()

        df.to_csv(URM_path, index=False, header=False)
        URM_path.seek(0)

        URM_all, URM_timestamp, item_original_ID_to_index, user_original_ID_to_index = load_CSV_into_SparseBuilder(URM_path, header=False, separator=",", timestamp=True, remove_duplicates=True)

        loaded_URM_dict = {"URM_all": URM_all,
                           "URM_timestamp": URM_timestamp}

        loaded_dataset = Dataset(dataset_name = self._get_dataset_name(),
                                 URM_dictionary = loaded_URM_dict,
                                 ICM_dictionary = None,
                                 ICM_feature_mapper_dictionary = None,
                                 UCM_dictionary = None,
                                 UCM_feature_mapper_dictionary = None,
                                 user_original_ID_to_index= user_original_ID_to_index,
                                 item_original_ID_to_index= item_original_ID_to_index,
                                 is_implicit = self.IS_IMPLICIT,
                                 )

        self._print("loading complete")

        return loaded_dataset