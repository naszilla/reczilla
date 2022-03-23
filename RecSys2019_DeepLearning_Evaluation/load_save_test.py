from pathlib import Path

import numpy as np
import sys
sys.path.append('..')

from Data_manager.CiaoDVD.CiaoDVDReader import CiaoDVDReader
from Data_manager.AmazonReviewData.AmazonAmazonInstantVideoReader import AmazonAmazonInstantVideoReader
from Data_manager.DataSplitter_leave_k_out import DataSplitter_leave_k_out
from Data_manager.DataSplitter import DataSplitter

data_reader = AmazonAmazonInstantVideoReader
data_reader = data_reader(folder=f"./test_load_save_{data_reader.DATASET_SUBFOLDER}")

loaded_dataset = data_reader.load_data()

dataSplitter = DataSplitter_leave_k_out(data_reader, k_out_value=1, use_validation_set=True, folder=f"./test_load_save_{data_reader.DATASET_SUBFOLDER}/split/")

dataSplitter.load_data()

# The code hereon loads a split without knowing its original splitter class and datareader class
dataReader_object, splitter_class, splitter_kwargs = DataSplitter.load_data_reader_splitter_class(Path(f"./test_load_save_{data_reader.DATASET_SUBFOLDER}/split/"))

newSplitter = splitter_class(dataReader_object, **splitter_kwargs)
