#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: xxxxx xxxxx
"""



from Data_manager.AmazonReviewData._AmazonReviewDataReader import _AmazonReviewDataReader


class AmazonToysGamesReader(_AmazonReviewDataReader):

    DATASET_URL_RATING = "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Toys_and_Games.csv"
    DATASET_URL_METADATA = "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Toys_and_Games.json.gz"

    DATASET_SUBFOLDER = "AmazonReviewData/AmazonToysGames/"
    AVAILABLE_ICM = ["ICM_metadata"]


    def _get_dataset_name_root(self):
        return self.DATASET_SUBFOLDER


    def _load_from_original_file(self):

        # Load data from original

        self._print("Loading original data")

        dataset_split_folder = self.DATASET_SPLIT_ROOT_FOLDER + self.DATASET_SUBFOLDER


        metadata_path = self._get_ICM_metadata_path(data_folder = dataset_split_folder,
                                                    compressed_file_name = "meta_Toys_and_Games.json.gz",
                                                    decompressed_file_name = "meta_Toys_and_Games.json",
                                                    file_url = self.DATASET_URL_METADATA)


        URM_path = self._get_URM_review_path(data_folder = dataset_split_folder,
                                            file_name = "ratings_Toys_and_Games.csv",
                                            file_url = self.DATASET_URL_RATING)


        loaded_dataset = self._load_from_original_file_all_amazon_datasets(URM_path,
                                                                        metadata_path = metadata_path,
                                                                        reviews_path = None)

        return loaded_dataset
    