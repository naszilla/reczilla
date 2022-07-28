#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 14/09/17

@author: Maurizio Ferrari Dacrema
"""


import zipfile, shutil
from Data_manager.Dataset import Dataset
from Data_manager.DataReader import DataReader
from Data_manager.DataReader_utils import download_from_URL, load_CSV_into_SparseBuilder
from Data_manager.IncrementalSparseMatrix import IncrementalSparseMatrix

import csv

class Movielens100KReader(DataReader):

    DATASET_URL = "http://files.grouplens.org/datasets/movielens/ml-100k.zip"
    DATASET_SUBFOLDER = "Movielens100K/"
    AVAILABLE_ICM = []

    IS_IMPLICIT = False

    def _get_dataset_name_root(self):
        return self.DATASET_SUBFOLDER


    def _load_from_original_file(self):
        # Load data from original

        zipFile_path =  self.DATASET_SPLIT_ROOT_FOLDER + self.DATASET_SUBFOLDER

        try:

            dataFile = zipfile.ZipFile(zipFile_path + "ml-100k.zip")

        except (FileNotFoundError, zipfile.BadZipFile):

            self._print("Unable to fild data zip file. Downloading...")

            download_from_URL(self.DATASET_URL, zipFile_path, "ml-100k.zip")

            dataFile = zipfile.ZipFile(zipFile_path + "ml-100k.zip")


        URM_path = dataFile.extract("ml-100k/u.data", path=zipFile_path + "decompressed/")

        URM_all, URM_timestamp, item_original_ID_to_index, user_original_ID_to_index = load_CSV_into_SparseBuilder(URM_path, separator="\t", header=False, timestamp=True)

        ICM_path = dataFile.extract("ml-100k/u.item", path=zipFile_path + "decompressed/")
        
        ICM_dict, ICM_feature_mapper_dict, self.item_name = self._load_ICM(ICM_path, item_original_ID_to_index)

        loaded_URM_dict = {"URM_all": URM_all,
                           "URM_timestamp": URM_timestamp}

        # Datareader object needs to know the ICM names 
        self.AVAILABLE_ICM = list(ICM_dict.keys())

        loaded_dataset = Dataset(dataset_name = self._get_dataset_name(),
                                 URM_dictionary = loaded_URM_dict,
                                 ICM_dictionary = ICM_dict,
                                 ICM_feature_mapper_dictionary = ICM_feature_mapper_dict,
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


# NOTE: See MovieLens README for item-features file schema
# https://files.grouplens.org/datasets/movielens/ml-100k-README.txt
    def _load_ICM(self, ICM_path, item_original_ID_to_index):
        ICM_builder = IncrementalSparseMatrix()

        # genres are stored as the last 19 columns in the tsv file
        genre_list = ['unknown', 'Action', 'Adventure', 'Animation',
            "Children's", 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
            'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
            'Thriller', 'War', 'Western']

        with open(ICM_path, 'r', encoding = 'ISO-8859-1') as rf:
            reader = csv.reader(rf, delimiter='|')
            item_name = dict()

            for row in reader:
                item_ID = row[0]
                genre_vector = row[-19:] 
                genres = [i for i,e in enumerate(genre_vector) if float(e) != 0]

                row_index = item_original_ID_to_index[item_ID]
                item_name[row_index] = row[1]
                ICM_builder.add_single_row(row_index, genres, data=1.0)


        ICM = ICM_builder.get_SparseMatrix()
        ICM_dict = {'ICM_genre': ICM}

        ICM_feature_mapper = {genre_name: idx for idx, genre_name in enumerate(genre_list)}
        ICM_feature_mapper_dict = {'ICM_genre': ICM_feature_mapper}
        
        return ICM_dict, ICM_feature_mapper_dict, item_name
                

