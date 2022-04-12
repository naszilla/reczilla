"""
Read a dataset split that has been created using DataSplitter instance methods save_data_reader_splitter_class() and
load_data(). These functions create one or more zip archives with the dataset split, and a pickle file with name ending
in "_class", that is used to initialize the DataSplitter class object.
"""

from Data_manager.DataSplitter import DataSplitter
from pathlib import Path

###############################################
# Step 1: read contents of the split directory.
###############################################

split_path = Path("./reczilla_examples/example_split")

(
    data_reader,
    splitter_class,
    init_kwargs,
) = DataSplitter.load_data_reader_splitter_class(split_path)


###############################################
# Step 2: initialize. the datasplitter object
###############################################

data_splitter = splitter_class(
    data_reader, folder=str(split_path), verbose=True,
)

# >>>>>> expected output:
#
# DataSplitter_leave_k_out: Cold users not allowed


###############################################
# Step 3: load the dataset split
###############################################

data_splitter.load_data()

# >>>>>> expected output:
#
# DataSplitter_leave_k_out: Verifying data consistency...
# DataSplitter_leave_k_out: Verifying data consistency... Passed!
# DataSplitter_leave_k_out: DataReader: Frappe
# 	Num items: 4082
# 	Num users: 777
# 	Train 		interactions 17022, 	density 5.37E-03
# 	Validation 	interactions 777, 	density 2.45E-04
# 	Test 		interactions 777, 	density 2.45E-04


###############################################
# Step 4: access the train/test/val URMs
###############################################

print(f"shape of the training URM: {data_splitter.SPLIT_URM_DICT['URM_train'].shape}")
print(f"number of nonzero elements: {data_splitter.SPLIT_URM_DICT['URM_train'].count_nonzero()}")

print(f"shape of the test URM: {data_splitter.SPLIT_URM_DICT['URM_test'].shape}")
print(f"number of nonzero elements: {data_splitter.SPLIT_URM_DICT['URM_test'].count_nonzero()}")

# >>>>>> expected output:
#
# shape of the training URM: (777, 4082)
# number of nonzero elements: 17022
# shape of the test URM: (777, 4082)
# number of nonzero elements: 777
