#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 01/01/2018

@author: Maurizio Ferrari Dacrema
"""

import os, traceback
from Data_manager.Dataset import Dataset


#################################################################################################################
#############################
#############################               DATA READER
#############################
#################################################################################################################



class DataReader(object):
    """
    Abstract class for the DataReaders, each shoud be implemented for a specific dataset
    DataReader has the following functions:
     - It loads the data of the original dataset and saves it into sparse matrices
     - It exposes the following functions
        - load_data(save_folder_path = None)        loads the data and saves into the specified folder, if None uses default, if False des not save
        - get_URM_all()                             returns a copy of the whole URM
        - get_ICM_from_name(ICM_name)               returns a copy of the specified ICM
        - get_loaded_ICM_names()                    returns a copy of the loaded ICM names, which can be used in get_ICM_from_name
        - get_loaded_ICM_dict()                     returns a copy of the loaded ICM in a dictionary [ICM_name]->ICM_sparse
        - DATASET_SUBFOLDER_DEFAULT                 path of the data folder
        - item_original_ID_to_index
        - user_original_ID_to_index

    """
    __DATASET_SPLIT_SUBFOLDER = "Data_manager_split_datasets/"
    __DATASET_OFFLINE_SUBFOLDER = "Data_manager_offline_datasets/"
    DATASET_SPLIT_ROOT_FOLDER = None
    DATASET_OFFLINE_ROOT_FOLDER = None

    # This subfolder contains the preprocessed data, already loaded from the original data file
    DATASET_SUBFOLDER_ORIGINAL = "original/"

    # Available URM split
    AVAILABLE_URM = ["URM_all"]

    # Available ICM for the given dataset, there might be no ICM, one or many
    AVAILABLE_ICM = []
    AVAILABLE_UCM = []

    # This flag specifies if the given dataset contains implicit preferences or explicit ratings
    IS_IMPLICIT = True

    _DATA_READER_NAME = "DataReader"

    def __init__(self,
                 reload_from_original_data="as-needed",  # {"always", "never", "as-needed"}
                 verbose=True,
                 folder=None,
                 ):
        super(DataReader, self).__init__()

        assert reload_from_original_data in ["as-needed", "always", "never"], f"invalid value of reload_from_original_data: {reload_from_original_data}"
        assert verbose in [True, False], "verbose parameter must be True or False"

        self.verbose = verbose
        self.folder = folder

        self.DATASET_SPLIT_ROOT_FOLDER = os.path.join(os.path.dirname(__file__), '..', self.__DATASET_SPLIT_SUBFOLDER)
        self.DATASET_OFFLINE_ROOT_FOLDER = os.path.join(os.path.dirname(__file__), '..', self.__DATASET_OFFLINE_SUBFOLDER)

        self.reload_from_original_data = reload_from_original_data
        if self.reload_from_original_data == "always":
            self._print("reload_from_original_data is 'always', previously loaded data will be ignored.")
        elif self.reload_from_original_data == "never":
            self._print("reload_from_original_data is 'never', will not reload original data.")
        else:
            self._print("reload_from_original_data is 'as-needed', will only reload original data if it cannot be found.")

    def _print(self, message):
        if self.verbose:
            print("{}: {}".format(self._get_dataset_name(), message))

    def _get_dataset_name(self):
        return self._get_dataset_name_root().replace("/", "_")[:-1]


    def get_loaded_ICM_names(self):
        return self.AVAILABLE_ICM.copy()


    def get_loaded_UCM_names(self):
        return self.AVAILABLE_UCM.copy()

    def _load_from_original_file(self):
        raise NotImplementedError("{}: _load_from_original_file was not implemented for the required dataset. Impossible to load the data".format(self._DATA_READER_NAME))


    def _get_dataset_name_root(self):
        """
        Returns the root of the folder tree which contains all of the dataset data/splits and files

        :return: Dataset_name/
        """
        raise NotImplementedError("{}:_get_dataset_name_root was not implemented for the required dataset. Impossible to load the data".format(self._DATA_READER_NAME))




    def _get_dataset_name_data_subfolder(self):
        """
        Returns the subfolder inside the dataset folder tree which contains the specific data to be loaded
        This method must be overridden by any data post processing object like k-cores / user sampling / interaction sampling etc
        to be applied before the data split

        :return: original or k_cores etc...
        """
        return self.DATASET_SUBFOLDER_ORIGINAL


    def load_data(self, save_folder_path = None):
        """
        :param save_folder_path:    path in which to save the loaded dataset
                                    None    use default "dataset_name/original/"
                                    False   do not save
        :return:
        """

        if save_folder_path is None:
            # use folder specified upon initialization
            if self.folder is not None:
                save_folder_path = self.folder
            else:
                # Use default e.g., "dataset_name/original/"
                save_folder_path = self.DATASET_SPLIT_ROOT_FOLDER + self._get_dataset_name_root() + self._get_dataset_name_data_subfolder()

        # this is extremely annoying
        if not save_folder_path.endswith(os.sep):
            save_folder_path = save_folder_path + os.sep

        # If save_folder_path contains any path try to load a previously built split from it
        if self.reload_from_original_data in ["as-needed", "always"] and save_folder_path in [None, False]:
            raise Exception("save_folder_path must be specified unless reload_from_original_data is 'never'")

        if self.reload_from_original_data in ["as-needed", "never"]:

            try:
                loaded_dataset = Dataset()
                loaded_dataset.load_data(save_folder_path)
                if not loaded_dataset.AVAILABLE_ICM:
                    self.AVAILABLE_ICM = loaded_dataset.AVAILABLE_ICM

                self._print("Verifying data consistency...")
                loaded_dataset.verify_data_consistency()
                self._print("Verifying data consistency... Passed!")

                # don't print statistics every time.
                # loaded_dataset.print_statistics()
                return loaded_dataset

            except FileNotFoundError:

                if self.reload_from_original_data == "never":
                    self._print("Preloaded data not found and reload_from_original_data = 'never'. Raising exception.")
                    raise FileNotFoundError
                else:
                    self._print("Preloaded data not found, reading from original files...")

            except Exception:

                self._print("Reading split from {} caused the following exception...".format(save_folder_path))
                traceback.print_exc()
                raise Exception("{}: Exception while reading split".format(self._get_dataset_name()))


        self._print("Loading original data")
        loaded_dataset = self._load_from_original_file()

        self._print("Verifying data consistency...")
        loaded_dataset.verify_data_consistency()
        self._print("Verifying data consistency... Passed!")

        if save_folder_path not in [False]:

            # If directory does not exist, create
            if not os.path.exists(save_folder_path):
                self._print("Creating folder '{}'".format(save_folder_path))
                os.makedirs(save_folder_path)

            else:
                self._print("Found already existing folder '{}'".format(save_folder_path))

            loaded_dataset.save_data(save_folder_path)

            self._print("Saving complete!")

        # don't print statistics every time
        # loaded_dataset.print_statistics()
        return loaded_dataset

# NOTE: this class is obsolete, will be removed in a future commit. instead, use function datareader_light
class GenericDataReader(DataReader):

    def _get_dataset_name(self):
        return self.__class__.__name__
