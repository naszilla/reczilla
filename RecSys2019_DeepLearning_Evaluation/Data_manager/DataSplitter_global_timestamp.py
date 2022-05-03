#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Maurizio Ferrari Dacrema and Sujay Khandagale
"""

from matplotlib import use
import scipy.sparse as sps
import numpy as np
import os
from Base.DataIO import DataIO

from Data_manager.DataSplitter import DataSplitter as _DataSplitter
from Data_manager.DataReader import DataReader as _DataReader

from Data_manager.DataReader_utils import compute_density, reconcile_mapper_with_removed_tokens
from Data_manager.split_functions.split_train_validation_leave_k_out import split_train_leave_k_out_user_wise
from Data_manager.split_functions.split_data_on_timestamp import split_data_on_timestamp
from Data_manager.split_functions.split_data_on_global_timestamp import split_data_on_global_timestamp
from Data_manager.data_consistency_check import assert_disjoint_matrices, assert_URM_ICM_mapper_consistency


class DataSplitter_global_timestamp(_DataSplitter):
    """
    The splitter tries to load from the specific folder related to a dataset, a split in the format corresponding to
    the splitter class. Basically each split is in a different subfolder
    - The "original" subfolder contains the whole dataset, is composed by a single URM with all data and may contain
        ICMs as well, either one or many, depending on the dataset
    - The other subfolders "warm", "cold" ecc contains the splitted data.

    The dataReader class involvement is limited to the following cased:
    - At first the dataSplitter tries to load from the subfolder corresponding to that split. Say "warm"
    - If the dataReader is succesful in loading the files, then a split already exists and the loading is complete
    - If the dataReader raises a FileNotFoundException, then no split is available.
    - The dataSplitter then creates a new instance of dataReader using default parameters, so that the original data will be loaded
    - At this point the chosen dataSplitter takes the URM_all and selected ICM to perform the split
    - The dataSplitter saves the splitted data in the appropriate subfolder.
    - Finally, the dataReader is instantiated again with the correct parameters, to load the data just saved
    """

    """
     - It exposes the following functions
        - load_data(save_folder_path = None, force_new_split = False)   loads the data or creates a new split
    
    
    """

    DATA_SPLITTER_NAME = "DataSplitter_global_timestamp"

    SPLIT_URM_DICT = None

    SPLIT_ICM_DICT = None
    SPLIT_ICM_MAPPER_DICT = None

    SPLIT_UCM_DICT = None
    SPLIT_UCM_MAPPER_DICT = None

    SPLIT_GLOBAL_MAPPER_DICT = None


    def __init__(self, dataReader_object:_DataReader, k_out_percent = 10, forbid_new_split = False, force_new_split = False, use_validation_set = True, leave_random_out = True, folder=None, verbose=True):
        """
        
        :param k_out_percent: Determines what percentile of timestamps go to test and eval. E.g when k_out_percent = 10, the 80th, 90th percentile timestamps are used to create train/val/test
        :param dataReader_object:
        :param n_folds:
        :param force_new_split:
        :param forbid_new_split:
        """


        assert use_validation_set == True, "Not using val set not supported at the moment."
        assert k_out_percent > 0 and k_out_percent < 49  # 2 * k_out_percent timestamps are reserved for test/eval
        
        self.k_out_percent = k_out_percent
        self.use_validation_set = use_validation_set
        self.allow_cold_users = False
        self.removed_cold_users = None
        self.leave_random_out = leave_random_out

        super(DataSplitter_global_timestamp, self).__init__(dataReader_object, forbid_new_split=forbid_new_split, force_new_split=force_new_split, folder=folder, verbose=verbose)

        self._print("Cold users not allowed")

        self.init_kwargs = {"k_out_percent": k_out_percent,
                            "forbid_new_split": forbid_new_split,
                            "force_new_split": force_new_split,
                            "use_validation_set": use_validation_set,
                            "leave_random_out": leave_random_out
                            }



    def _get_split_subfolder_name(self):
        """

        :return: warm_{n_folds}_fold/
        """

        if self.leave_random_out:
            order_suffix = "random"
        else:
            order_suffix = "last"


        return "leave_{}_out_{}/".format(self.k_out_percent, order_suffix)


    def get_statistics_URM(self):


        self._assert_is_initialized()

        n_users, n_items = self.SPLIT_URM_DICT["URM_train"].shape

        statistics_string = "DataReader: {}\n" \
                            "\tNum items: {}\n" \
                            "\tNum users: {}\n" \
                            "\tTrain \t\tinteractions {}, \tdensity {:.2E}\n".format(
                            self.dataReader_object._get_dataset_name(),
                            n_items,
                            n_users,
                            self.SPLIT_URM_DICT["URM_train"].nnz, compute_density(self.SPLIT_URM_DICT["URM_train"]))

        if self.use_validation_set:
            statistics_string += "\tValidation \tinteractions {}, \tdensity {:.2E}\n".format(
                                    self.SPLIT_URM_DICT["URM_validation"].nnz, compute_density(self.SPLIT_URM_DICT["URM_validation"]))


        statistics_string += "\tTest \t\tinteractions {}, \tdensity {:.2E}\n".format(
                                self.SPLIT_URM_DICT["URM_test"].nnz, compute_density(self.SPLIT_URM_DICT["URM_test"]))


        self._print(statistics_string)

        self._print("\n")




    def get_ICM_from_name(self, ICM_name):
        return self.SPLIT_ICM_DICT[ICM_name].copy()


    def get_statistics_ICM(self):

        self._assert_is_initialized()

        if len(self.dataReader_object.get_loaded_ICM_names())>0:

            for ICM_name, ICM_object in self.SPLIT_ICM_DICT.items():

                n_items, n_features = ICM_object.shape

                statistics_string = "\tICM name: {}, Num features: {}, feature occurrences: {}, density {:.2E}".format(
                    ICM_name,
                    n_features,
                    ICM_object.nnz,
                    compute_density(ICM_object)
                )

                self._print(statistics_string)

            self._print("\n")



    def _assert_is_initialized(self):
         assert self.SPLIT_URM_DICT is not None, "{}: Unable to load data split. The split has not been generated yet, call the load_data function to do so.".format(self.DATA_SPLITTER_NAME)


    def get_holdout_split(self):
        """
        The train set is defined as all data except the one of that fold, which is the test
        :return: URM_train, URM_validation, URM_test
        """

        self._assert_is_initialized()

        if self.use_validation_set:
            return self.SPLIT_URM_DICT["URM_train"].copy(),\
                   self.SPLIT_URM_DICT["URM_validation"].copy(),\
                   self.SPLIT_URM_DICT["URM_test"].copy()

        return self.SPLIT_URM_DICT["URM_train"].copy(), self.SPLIT_URM_DICT["URM_test"].copy()


    def _split_data_from_original_dataset(self, save_folder_path):

        self.loaded_dataset = self.dataReader_object.load_data()
        self._load_from_DataReader_ICM_and_mappers(self.loaded_dataset)

        URM = self.loaded_dataset.get_URM_all()
        URM = sps.csr_matrix(URM)

        URM_timestamp = self.loaded_dataset.get_URM_timestamp()

        split_number = 2
        if self.use_validation_set:
            split_number+=1
        
        URM_train, URM_validation, URM_test, users_to_remove = split_data_on_global_timestamp(URM, URM_timestamp, self.k_out_percent)
        
        if not self.allow_cold_users:  # always satisfied
            user_to_preserve = np.logical_not(users_to_remove)
            self.removed_cold_users = users_to_remove

            URM = URM[user_to_preserve,:]
            

            self.SPLIT_GLOBAL_MAPPER_DICT["user_original_ID_to_index"] = reconcile_mapper_with_removed_tokens(self.SPLIT_GLOBAL_MAPPER_DICT["user_original_ID_to_index"],
                                                                                                              np.arange(0, len(self.removed_cold_users), dtype=np.int)[self.removed_cold_users])

            for UCM_name, UCM_object in self.SPLIT_UCM_DICT.items():
                UCM_object = UCM_object[user_to_preserve,:]
                self.SPLIT_UCM_DICT[UCM_name] = UCM_object


        self.SPLIT_URM_DICT = {
            "URM_train": URM_train,
            "URM_test": URM_test,
        }

        # ensure atleast 10 entries in train and test splits
        assert URM_train.nnz > 10 and URM_test.nnz > 10, f"{URM_train.nnz} entries in train, {URM_test.nnz} entries in test splits"

        if self.use_validation_set:
            self.SPLIT_URM_DICT["URM_validation"] = URM_validation

        self._save_split(save_folder_path)

        self._print("Split complete")




    def _save_split(self, save_folder_path):

        if save_folder_path:
            
            if not os.path.exists(save_folder_path):
                os.makedirs(save_folder_path)

            self.save_data_reader_splitter_class(save_folder_path)

            if self.allow_cold_users:
                allow_cold_users_suffix = "allow_cold_users"

            else:
                allow_cold_users_suffix = "only_warm_users"

            if self.use_validation_set:
                validation_set_suffix = "use_validation_set"
            else:
                validation_set_suffix = "no_validation_set"


            name_suffix = "_{}_{}".format(allow_cold_users_suffix, validation_set_suffix)

            split_parameters_dict = {"k_out_percent": self.k_out_percent,
                                     "allow_cold_users": self.allow_cold_users,
                                     "removed_cold_users": self.removed_cold_users,
                                     }



            dataIO = DataIO(folder_path = save_folder_path)

            dataIO.save_data(data_dict_to_save = split_parameters_dict,
                             file_name = "split_parameters" + name_suffix)

            dataIO.save_data(data_dict_to_save = self.SPLIT_GLOBAL_MAPPER_DICT,
                             file_name = "split_mappers" + name_suffix)

            dataIO.save_data(data_dict_to_save = self.SPLIT_URM_DICT,
                             file_name = "split_URM" + name_suffix)

            if len(self.SPLIT_ICM_DICT)>0:
                dataIO.save_data(data_dict_to_save = self.SPLIT_ICM_DICT,
                                 file_name = "split_ICM" + name_suffix)

                dataIO.save_data(data_dict_to_save = self.SPLIT_ICM_MAPPER_DICT,
                                 file_name = "split_ICM_mappers" + name_suffix)


            if len(self.SPLIT_UCM_DICT)>0:
                dataIO.save_data(data_dict_to_save = self.SPLIT_UCM_DICT,
                                 file_name = "split_UCM" + name_suffix)

                dataIO.save_data(data_dict_to_save = self.SPLIT_UCM_MAPPER_DICT,
                                 file_name = "split_UCM_mappers" + name_suffix)


    def _load_previously_built_split_and_attributes(self, save_folder_path):
        """
        Loads all URM and ICM
        :return:
        """


        if self.use_validation_set:
            validation_set_suffix = "use_validation_set"
        else:
            validation_set_suffix = "no_validation_set"

        if self.allow_cold_users:
            allow_cold_users_suffix = "allow_cold_users"
        else:
            allow_cold_users_suffix = "only_warm_users"


        name_suffix = "_{}_{}".format(allow_cold_users_suffix, validation_set_suffix)


        dataIO = DataIO(folder_path = save_folder_path)

        split_parameters_dict = dataIO.load_data(file_name ="split_parameters" + name_suffix)

        for attrib_name in split_parameters_dict.keys():
             self.__setattr__(attrib_name, split_parameters_dict[attrib_name])


        self.SPLIT_GLOBAL_MAPPER_DICT = dataIO.load_data(file_name ="split_mappers" + name_suffix)

        self.SPLIT_URM_DICT = dataIO.load_data(file_name ="split_URM" + name_suffix)

        if len(self.dataReader_object.get_loaded_ICM_names())>0:
            self.SPLIT_ICM_DICT = dataIO.load_data(file_name ="split_ICM" + name_suffix)

            self.SPLIT_ICM_MAPPER_DICT = dataIO.load_data(file_name ="split_ICM_mappers" + name_suffix)


        if len(self.dataReader_object.get_loaded_UCM_names())>0:
            self.SPLIT_UCM_DICT = dataIO.load_data(file_name ="split_UCM" + name_suffix)

            self.SPLIT_UCM_MAPPER_DICT = dataIO.load_data(file_name ="split_UCM_mappers" + name_suffix)

    #########################################################################################################
    ##########                                                                                     ##########
    ##########                                DATA CONSISTENCY                                     ##########
    ##########                                                                                     ##########
    #########################################################################################################


    def _verify_data_consistency(self):

        self._assert_is_initialized()

        print_preamble = "{} consistency check: ".format(self.DATA_SPLITTER_NAME)

        URM_to_load_list = ["URM_train", "URM_test"]

        if self.use_validation_set:
            URM_to_load_list.append("URM_validation")


        assert len(self.SPLIT_URM_DICT) == len(URM_to_load_list),\
            print_preamble + "The available URM are not as many as they are supposed to be. URMs are {}, expected URMs are {}".format(len(self.SPLIT_URM_DICT), len(URM_to_load_list))


        assert all(URM_name in self.SPLIT_URM_DICT for URM_name in URM_to_load_list), print_preamble + "Not all URMs have been created"
        assert all(URM_name in URM_to_load_list for URM_name in self.SPLIT_URM_DICT.keys()), print_preamble + "The split contains URMs that should not exist"


        URM_shape = None

        for URM_name, URM_object in self.SPLIT_URM_DICT.items():

            if URM_shape is None:
                URM_shape = URM_object.shape

                n_users, n_items = URM_shape

                assert n_users != 0,  print_preamble + "Number of users in URM is 0"
                assert n_items != 0,  print_preamble + "Number of items in URM is 0"

            assert URM_shape == URM_object.shape,  print_preamble + "URM shape is inconsistent"


        assert self.SPLIT_URM_DICT["URM_train"].nnz != 0, print_preamble + "Number of interactions in URM Train is 0"
        assert self.SPLIT_URM_DICT["URM_test"].nnz != 0, print_preamble + "Number of interactions in URM Test is 0"


        URM = self.SPLIT_URM_DICT["URM_test"].copy()
        user_interactions = np.ediff1d(sps.csr_matrix(URM).indptr)

        # assert np.all(user_interactions == self.k_out_value), print_preamble + "Not all users have the desired number of interactions in URM_test, {} users out of {}".format(
        #     (user_interactions != self.k_out_value).sum(), n_users)



        if self.use_validation_set:
            assert self.SPLIT_URM_DICT["URM_validation"].nnz != 0, print_preamble + "Number of interactions in URM Validation is 0"

            URM = self.SPLIT_URM_DICT["URM_validation"].copy()
            user_interactions = np.ediff1d(sps.csr_matrix(URM).indptr)

            # assert np.all(user_interactions == self.k_out_value), print_preamble + "Not all users have the desired number of interactions in URM_validation, {} users out of {}".format(
            #     (user_interactions != self.k_out_value).sum(), n_users)



        URM = self.SPLIT_URM_DICT["URM_train"].copy()
        user_interactions = np.ediff1d(sps.csr_matrix(URM).indptr)

        if not self.allow_cold_users:
            assert np.all(user_interactions != 0), print_preamble + "Cold users exist despite not being allowed as per DataSplitter parameters, {} users out of {}".format(
                (user_interactions == 0).sum(), n_users)


        assert assert_disjoint_matrices(list(self.SPLIT_URM_DICT.values()))

        assert_URM_ICM_mapper_consistency(URM_DICT = self.SPLIT_URM_DICT,
                                          user_original_ID_to_index=self.SPLIT_GLOBAL_MAPPER_DICT["user_original_ID_to_index"],
                                          item_original_ID_to_index=self.SPLIT_GLOBAL_MAPPER_DICT["item_original_ID_to_index"],
                                          ICM_DICT = self.SPLIT_ICM_DICT,
                                          ICM_MAPPER_DICT = self.SPLIT_ICM_MAPPER_DICT,
                                          UCM_DICT = self.SPLIT_UCM_DICT,
                                          UCM_MAPPER_DICT = self.SPLIT_UCM_MAPPER_DICT,
                                          DATA_SPLITTER_NAME = self.DATA_SPLITTER_NAME)

