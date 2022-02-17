#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Sujay Khandagale
"""

import gzip, shutil
import pandas as pd
from io import StringIO
import json
from Data_manager.Dataset import Dataset
from Data_manager.DataReader import DataReader
from Data_manager.DataReader_utils import download_from_URL, load_CSV_into_SparseBuilder


class GoogleLocalReviewsReader(DataReader):

    DATASET_URL = "http://deepyeti.ucsd.edu/jmcauley/datasets/googlelocal/reviews.clean.json.gz"
    DATASET_SUBFOLDER = "GoogleLocalReviews/"
    AVAILABLE_ICM = []
    DATASET_SPECIFIC_MAPPER = []

    IS_IMPLICIT = False


    def _get_dataset_name_root(self):
        return self.DATASET_SUBFOLDER
    
    def _load_df(self, gz_path):
        final_dict = {'user_id': [], 'item_id': [], 'rating': [], 'timestamp': []}
        
        with gzip.GzipFile(gz_path, "r") as f:
                for line in f:
                    dic = eval(line)
                    user_id, item_id, rating, timestamp = dic['gPlusUserId'], dic['gPlusPlaceId'], dic['rating'], dic['unixReviewTime']
                    final_dict['user_id'].append(user_id)
                    final_dict['item_id'].append(item_id)
                    final_dict['rating'].append(rating)
                    final_dict['timestamp'].append(timestamp)


        return pd.DataFrame.from_dict(final_dict)




    def _load_from_original_file(self):
        # Load data from original

        self._print("Loading original data")

        zipFile_path =  self.DATASET_SPLIT_ROOT_FOLDER + self.DATASET_SUBFOLDER

        try:
            # df = pd.read_json(zipFile_path + "reviews.clean.json.gz", lines=True, compression='gzip', orient='split', encoding='utf-8-sig')
            df = self._load_df(zipFile_path + "reviews.clean.json.gz")

        except (FileNotFoundError, ):

            print("GoogleLocalReviews: Unable to fild data zip file. Downloading...")

            download_from_URL(self.DATASET_URL, zipFile_path, "reviews.clean.json.gz")

            df = self._load_df(zipFile_path + "reviews.clean.json.gz")

        URM_path = StringIO()

        df.to_csv(URM_path, index=False)
        URM_path.seek(0)

        URM_all, URM_timestamp, item_original_ID_to_index, user_original_ID_to_index = load_CSV_into_SparseBuilder(URM_path, header=True, separator=",", timestamp=True, remove_duplicates=True)

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


        self._print("cleaning temporary files")

        shutil.rmtree(zipFile_path + "decompressed", ignore_errors=True)

        self._print("loading complete")

        return loaded_dataset