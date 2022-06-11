#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: xxxxx xxxxx
"""

import csv
from io import StringIO
from unittest import skip
import pandas as pd
import zipfile, shutil
from Data_manager.Dataset import Dataset
from Data_manager.DataReader import DataReader
from Data_manager.DataReader_utils import download_from_URL, load_CSV_into_SparseBuilder


class FlixsterReader(DataReader):

    DATASET_URL = "http://datasets.syr.edu/uploads/1296675547/Flixster-dataset.zip"
    DATASET_SUBFOLDER = "Flixster/"
    AVAILABLE_ICM = []
    DATASET_SPECIFIC_MAPPER = []

    IS_IMPLICIT = True


    def _get_dataset_name_root(self):
        return self.DATASET_SUBFOLDER



    def _load_from_original_file(self):
        # Load data from original

        self._print("Loading original data")

        zipFile_path =  self.DATASET_SPLIT_ROOT_FOLDER + self.DATASET_SUBFOLDER

        try:

            dataFile = zipfile.ZipFile(zipFile_path + "Flixster-dataset.zip")

        except (FileNotFoundError, zipfile.BadZipFile):

            print("Flixster: Unable to fild data zip file. Downloading...")

            download_from_URL(self.DATASET_URL, zipFile_path, "Flixster-dataset.zip")

            dataFile = zipfile.ZipFile(zipFile_path + "Flixster-dataset.zip")

        URM_path = dataFile.extract("Flixster-dataset/data/edges.csv", path=zipFile_path + "decompressed/")

        df = pd.read_csv(zipFile_path + "decompressed/" + "Flixster-dataset/data/edges.csv", skiprows=1, header=None)

        df.columns = ['user_id', 'item_id']
        df['rating'] = 1

        URM_path = StringIO()

        df.to_csv(URM_path, index=False)
        URM_path.seek(0)

        URM_all, item_original_ID_to_index, user_original_ID_to_index = load_CSV_into_SparseBuilder(URM_path, separator=",", header=True, remove_duplicates=True)

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