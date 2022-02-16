#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Sujay Khandagale
"""

import tarfile, os, shutil
from Data_manager.Dataset import Dataset
from Data_manager.DataReader import DataReader
from Data_manager.DataReader_utils import download_from_URL, load_CSV_into_SparseBuilder


class CiaoDVDReader(DataReader):

    DATASET_URL = "http://konect.cc/files/download.tsv.librec-ciaodvd-movie_ratings.tar.bz2"
    DATASET_SUBFOLDER = "CiaoDVD/"
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

            dataFile = tarfile.open(tarFile_path + "download.tsv.librec-ciaodvd-movie_ratings.tar.bz2", "r:bz2")

        except (FileNotFoundError, tarfile.TarError):

            print("CiaoDVD: Unable to fild data zip file. Downloading...")

            download_from_URL(self.DATASET_URL, tarFile_path, "download.tsv.librec-ciaodvd-movie_ratings.tar.bz2")

            dataFile = tarfile.open(tarFile_path + "download.tsv.librec-ciaodvd-movie_ratings.tar.bz2", "r:bz2")


        URM_path = dataFile.extractfile("librec-ciaodvd-movie_ratings/out.librec-ciaodvd-movie_ratings")

        URM_all, URM_timestamp, item_original_ID_to_index, user_original_ID_to_index = load_CSV_into_SparseBuilder(URM_path, header=False, separator=r"\s+", timestamp=True, remove_duplicates=True, skiprows=1)

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

        shutil.rmtree(tarFile_path + "decompressed", ignore_errors=True)

        self._print("loading complete")

        return loaded_dataset