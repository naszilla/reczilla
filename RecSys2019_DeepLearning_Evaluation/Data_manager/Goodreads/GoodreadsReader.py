#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: xxxxx xxxxx
"""

import shutil, os
from io import StringIO
import pandas as pd
from Data_manager.Dataset import Dataset
from Data_manager.DataReader import DataReader
from Data_manager.DataReader_utils import download_from_URL, load_CSV_into_SparseBuilder


class GoodreadsReader(DataReader):

    DATASET_URL = "https://figshare.com/ndownloader/files/34038896?private_link=2fa4daf07bcda933fce8"
    DATASET_SUBFOLDER = "Goodreads/"
    AVAILABLE_ICM = []
    DATASET_SPECIFIC_MAPPER = []

    IS_IMPLICIT = False


    def _get_dataset_name_root(self):
        return self.DATASET_SUBFOLDER



    def _load_from_original_file(self):
        # Load data from original

        self._print("Loading original data")

        csv_path =  self.DATASET_SPLIT_ROOT_FOLDER + self.DATASET_SUBFOLDER

        try:

            df = pd.read_csv(csv_path + "goodreads_interactions.csv")

        except:

            print("Goodreads: Unable to find data csv file. Downloading...")

            if not os.path.exists(csv_path):
                os.makedirs(csv_path)

            download_from_URL(self.DATASET_URL, csv_path, "goodreads_interactions.csv")
            df = pd.read_csv(csv_path + "goodreads_interactions.csv")

        df = df[['user_id', 'book_id', 'rating']]
        df.columns = ['user_id', 'item_id', 'rating']

        URM_path = StringIO()

        df.to_csv(URM_path, index=False)
        URM_path.seek(0)

        URM_all, item_original_ID_to_index, user_original_ID_to_index = load_CSV_into_SparseBuilder(URM_path, header=True, separator=",", timestamp=False, remove_duplicates=True)

        loaded_URM_dict = {"URM_all": URM_all}

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


        self._print("cleaning temporary files")

        shutil.rmtree(csv_path, ignore_errors=True)

        self._print("loading complete")

        return loaded_dataset