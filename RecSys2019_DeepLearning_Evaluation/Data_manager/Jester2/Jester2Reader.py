#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Sujay Khandagale
"""

import zipfile, shutil
from Data_manager.Dataset import Dataset
from Data_manager.DataReader import DataReader
from Data_manager.DataReader_utils import download_from_URL, load_CSV_into_SparseBuilder


class Jester2Reader(DataReader):

    DATASET_URL = "http://eigentaste.berkeley.edu/dataset/archive/jester_dataset_2.zip"
    DATASET_SUBFOLDER = "Jester2/"
    AVAILABLE_ICM = []
    DATASET_SPECIFIC_MAPPER = []

    IS_IMPLICIT = False


    def _get_dataset_name_root(self):
        return self.DATASET_SUBFOLDER



    def _load_from_original_file(self):
        # Load data from original

        self._print("Loading original data")

        zipFile_path =  self.DATASET_SPLIT_ROOT_FOLDER + self.DATASET_SUBFOLDER

        try:

            dataFile = zipfile.ZipFile(zipFile_path + "jester_dataset_2.zip")

        except (FileNotFoundError, zipfile.BadZipFile):

            print("Jester2: Unable to fild data zip file. Downloading...")

            download_from_URL(self.DATASET_URL, zipFile_path, "jester_dataset_2.zip")

            dataFile = zipfile.ZipFile(zipFile_path + "jester_dataset_2.zip")


        URM_path = dataFile.extract("jester_ratings.dat", path=zipFile_path + "decompressed/")

        URM_all, item_original_ID_to_index, user_original_ID_to_index = load_CSV_into_SparseBuilder(URM_path, separator="\t\t", header=False, remove_duplicates=True)

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

        shutil.rmtree(zipFile_path + "decompressed", ignore_errors=True)

        self._print("loading complete")

        return loaded_dataset