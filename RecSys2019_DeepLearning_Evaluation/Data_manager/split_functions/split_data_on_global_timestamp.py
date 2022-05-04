#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Sujay Khandagale
"""

import numpy as np
import scipy.sparse as sps

from Data_manager.IncrementalSparseMatrix import IncrementalSparseMatrix



def split_data_on_global_timestamp(URM_all, URM_timestamp, ts_val, ts_test):

    URM_all = sps.csr_matrix(URM_all)
    URM_timestamp = sps.csr_matrix(URM_timestamp)

    n_rows, n_cols = URM_all.shape


    URM_train_builder = IncrementalSparseMatrix(n_rows=n_rows, n_cols=n_cols)
    URM_test_builder = IncrementalSparseMatrix(n_rows=n_rows, n_cols=n_cols)
    URM_validation_builder = IncrementalSparseMatrix(n_rows=n_rows, n_cols=n_cols)

    all_items = np.arange(0, n_cols, dtype=np.int)
    skipped_users = 0

    for user_index in range(URM_all.shape[0]):

        # if user_index % 10000 == 0:
        #     print("split_data_on_global_timestamp: user {} of {}".format(user_index, URM_all.shape[0]))

        start_pos = URM_all.indptr[user_index]
        end_pos = URM_all.indptr[user_index+1]

        user_profile = URM_all.indices[start_pos:end_pos]
        user_data = URM_all.data[start_pos:end_pos]
        user_sequence = URM_timestamp.data[start_pos:end_pos]


        # if len(user_profile) >= 3:
        test_pos = np.where(user_sequence >= ts_test, True, False)
        val_pos = np.where((user_sequence >= ts_val) & (user_sequence < ts_test), True, False)
        train_pos = np.where(user_sequence < ts_val, True, False)
        
        if not any(train_pos):
            skipped_users += 1
            continue # do not include cold start users
        
        # test
        test_user_indices = [user_index] * sum(test_pos)
        test_venue_indices = user_profile[test_pos]
        test_venue_data = user_data[test_pos]
        URM_test_builder.add_data_lists(test_user_indices, test_venue_indices, test_venue_data)
        
        # eval
        val_user_indices = [user_index] * sum(val_pos)
        val_venue_indices = user_profile[val_pos]
        val_venue_data = user_data[val_pos]
        URM_validation_builder.add_data_lists(val_user_indices, val_venue_indices, val_venue_data)
        
        # train
        train_user_indices = [user_index] * sum(train_pos)
        train_venue_indices = user_profile[train_pos]
        train_venue_data = user_data[train_pos]
        URM_train_builder.add_data_lists(train_user_indices, train_venue_indices, train_venue_data)
            


            # URM_train_builder.add_data_lists([user_index]*len(user_profile), user_profile, user_data)


    URM_train = URM_train_builder.get_SparseMatrix()
    URM_validation = URM_validation_builder.get_SparseMatrix()
    URM_test = URM_test_builder.get_SparseMatrix()
    
    print(f"split_data_on_global_timestamp: {skipped_users} cold users of total {URM_all.shape[0]} users skipped")

    return URM_train, URM_validation, URM_test


