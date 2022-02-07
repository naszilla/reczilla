"""
These tests attempt to load each dataset in the dataset_handler.
- If DOWNLOAD_DATA=True, each dataset is downloaded to DATA_DIR, and basic unittests are run for each.
- If DOWNLOAD_DATA=False, we attempt to load each dataset from DATA_DIR. If a dataset is found, then we run basic
    unittests. If a dataset is not found, then the test passes.
"""

import unittest

from dataset_handler import DATASET_READER_LIST
import os

# if True, download the data if it is not found.
# if False, only run the test assertions if the dataset is found.
DOWNLOAD_DATA = False

# dir where the tests will search for each dataset
# if DOWNLOAD_DATA = True, datasets will be downloaded here if not found),
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


class TestDataReaders(unittest.TestCase):
    """
    NOTE: this test uses subtests to iterate over each dataset. it's a good idea to run this from command line, because
    the IDE might not use the correct configuration:
    > python -m tests.datareader_test
    """

    def test_datareaders(self):
        for i, reader_class in enumerate(DATASET_READER_LIST):
            with self.subTest(reader_name=reader_class.__name__):
                self.assertTrue(reader_class in DATASET_READER_LIST)
                temp_data_folder = os.path.join(DATA_DIR, reader_class.__name__) + "/"
                print("TEMP DATA FOLDER:")
                print(temp_data_folder)
                # attempt to load the dataset
                if DOWNLOAD_DATA:
                    try:
                        data_reader = reader_class(
                            reload_from_original_data="as-needed"
                        )
                        loaded_dataset = data_reader.load_data(
                            save_folder_path=temp_data_folder
                        )
                    except Exception as e:
                        self.fail(f"exception raised: {e}")

                else:
                    try:
                        data_reader = reader_class(reload_from_original_data="never")
                        loaded_dataset = data_reader.load_data(
                            save_folder_path=temp_data_folder
                        )
                    except FileNotFoundError:
                        print(
                            f"dataset not found in {temp_data_folder}. skipping this test."
                        )
                        return

                # make sure that URM_all is available and has positive dimensions
                try:
                    URM_all = loaded_dataset.get_URM_all()

                    self.assertTrue(URM_all.shape[0] > 0)
                    self.assertTrue(URM_all.shape[1] > 0)

                except Exception as e:
                    self.fail(f"exception raised: {e}")


if __name__ == "__main__":
    unittest.main()
