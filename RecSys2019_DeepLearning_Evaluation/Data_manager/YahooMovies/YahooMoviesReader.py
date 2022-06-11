#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Sujay Khandagale


TODO: Using only the train file for now. Maybe add test in the future
"""

import tarfile, os, shutil, gzip
from Data_manager.Dataset import Dataset
from Data_manager.DataReader import DataReader
from Data_manager.DataReader_utils import download_from_URL, load_CSV_into_SparseBuilder


class YahooMoviesReader(DataReader):

    DATASET_URL = "https://figshare.com/ndownloader/files/34939932?private_link=5ec186df491d4701d1f7"
    DATASET_SUBFOLDER = "YahooMovies/"
    AVAILABLE_ICM = []
    DATASET_SPECIFIC_MAPPER = []

    IS_IMPLICIT = False


    def _get_dataset_name_root(self):
        return self.DATASET_SUBFOLDER



    def _load_from_original_file(self):
        # Load data from original

        self._print("Loading original data")

        tarFile_path =  self.DATASET_SPLIT_ROOT_FOLDER + self.DATASET_SUBFOLDER

        try:

            dataFile = tarfile.open(tarFile_path + "download.yahooMovies-ratings.tgz", "r:gz")

        except (FileNotFoundError, tarfile.TarError):

            print("YahooMovies: Unable to fild data zip file. Downloading...")

            download_from_URL(self.DATASET_URL, tarFile_path, "download.yahooMovies-ratings.tgz")

            dataFile = tarfile.open(tarFile_path + "download.yahooMovies-ratings.tgz", "r:gz")
        
        dataFile.extractall(tarFile_path + 'decompressed')

        URM_path = gzip.open(tarFile_path + "decompressed/" + "ydata-ymovies-user-movie-ratings-train-v1_0.txt.gz", 'rb')

        URM_all, item_original_ID_to_index, user_original_ID_to_index = load_CSV_into_SparseBuilder(URM_path, separator="\t", header=False, remove_duplicates=True, custom_user_item_rating_columns=[0, 1, 3])

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

        shutil.rmtree(tarFile_path + "decompressed", ignore_errors=True)

        self._print("loading complete")

        return loaded_dataset