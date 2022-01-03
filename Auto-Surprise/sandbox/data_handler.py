from surprise import Dataset, Reader
from urllib.request import urlopen
from zipfile import ZipFile
import pandas as pd
import os

INBUILT_DATASETS = ['ml-100k', 'ml-1m', 'jester']
CUSTOM_DATASETS = ['book-crossing', 'dating']
TMP_DATASET_DOWNLOAD_DIR = ['data/']

def get_book_crossing():
    zipurl = 'http://www2.informatik.uni-freiburg.de/~cziegler/BX/BX-CSV-Dump.zip'
    zipresp = urlopen(zipurl)
    filename = "data/book-crossing/BX-CSV-Dump.zip"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    tempzip = open(filename, "wb")
    tempzip.write(zipresp.read())
    tempzip.close()
    zf = ZipFile("data/book-crossing/BX-CSV-Dump.zip")
    zf.extractall(path = 'data/book-crossing/')
    zf.close()

    # Load Book crossing dataset
    df = pd.read_csv('data/book-crossing/BX-Book-Ratings.csv', sep=';', error_bad_lines=False, encoding="latin-1")
    df.columns = ['user', 'item', 'rating']

    reader = Reader(rating_scale=(0, 10))
    data = Dataset.load_from_df(df.sample(n=100000, random_state=7), reader=reader)
    return data

def get_dating():
    df = pd.read_csv('data/dating/ratings.dat', sep=',', error_bad_lines=False, encoding="latin-1", header=None)
    df.columns = ['user', 'item', 'rating']

    reader = Reader(rating_scale=(0, 10))
    data = Dataset.load_from_df(df.sample(n=100000, random_state=7), reader=reader)
    return data

def get_custom_dataset(dataset_name: str):
    if dataset_name == 'book-crossing':
        return get_book_crossing()
    elif dataset_name == 'dating':
        return get_dating()
    else:
        raise NotImplementedError(f'Datasets currently supported are: {INBUILT_DATASETS + CUSTOM_DATASETS}')

def get_data(dataset_name: str):
    if dataset_name in INBUILT_DATASETS:
        return Dataset.load_builtin(dataset_name)
    return get_custom_dataset(dataset_name)

