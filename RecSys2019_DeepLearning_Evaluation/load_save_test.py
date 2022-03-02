import numpy as np
import sys
sys.path.append('..')

from Data_manager.CiaoDVD.CiaoDVDReader import CiaoDVDReader
from Data_manager.DataSplitter_leave_k_out import DataSplitter_leave_k_out
from Data_manager.DataSplitter import DataSplitter

data_reader = CiaoDVDReader()

loaded_dataset = data_reader.load_data(save_folder_path=f"./test_load_save_{data_reader.DATASET_SUBFOLDER}")

dataSplitter = DataSplitter_leave_k_out(data_reader, k_out_value=1, use_validation_set=False)

dataSplitter.load_data(save_folder_path=f"./test_load_save_{data_reader.DATASET_SUBFOLDER}")

# The code hereon loads a split without knowing its original splitter class

splitter_class, splitter_kwargs = DataSplitter.load_data_splitter_class(f"./test_load_save_{data_reader.DATASET_SUBFOLDER}")

newSplitter = splitter_class(data_reader, splitter_kwargs)
