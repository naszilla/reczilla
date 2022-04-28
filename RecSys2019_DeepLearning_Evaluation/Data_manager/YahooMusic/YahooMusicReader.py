#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Sujay Khandagale


TODO: Handle the 255 rating case (Don't want to see again)
"""

import tarfile, os, shutil, gzip
from Data_manager.Dataset import Dataset
from Data_manager.DataReader import DataReader
from Data_manager.DataReader_utils import download_from_URL, load_CSV_into_SparseBuilder


class YahooMusicReader(DataReader):

    DATASET_URL = "https://figshare.com/ndownloader/files/34938798?private_link=526257f8d2beb94d3a2e"
    DATASET_SUBFOLDER = "YahooMusic/"
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

            dataFile = tarfile.open(tarFile_path + "download.yahooMusic-ratings.tgz", "r:gz")

        except (FileNotFoundError, tarfile.TarError):

            print("Wikilens: Unable to fild data zip file. Downloading...")

            download_from_URL(self.DATASET_URL, tarFile_path, "download.yahooMusic-ratings.tgz")

            dataFile = tarfile.open(tarFile_path + "download.yahooMusic-ratings.tgz", "r:gz")
        
        dataFile.extractall(tarFile_path + 'decompressed')

        URM_path = gzip.open(tarFile_path + "decompressed/" + "ydata-ymusic-user-artist-ratings-v1_0.txt.gz", 'rb')

        URM_all, item_original_ID_to_index, user_original_ID_to_index = load_CSV_into_SparseBuilder(URM_path, separator="\t", header=False, remove_duplicates=True)

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