from surprise import Dataset, Reader
from urllib.request import urlopen
from zipfile import ZipFile
import pandas as pd
import os
import requests

INBUILT_DATASETS = ['ml-100k', 'ml-1m', 'jester']
CUSTOM_DATASETS = ['book-crossing', 'dating', 'recipes', 'amazon-reviews-all', 'amazon-reviews-movies', 'amazon-reviews-books', 'twitter-movie-ratings']
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
    print('shape', df.shape)

    reader = Reader(rating_scale=(0, 10))
    data = Dataset.load_from_df(df.sample(n=100000, random_state=7), reader=reader)
    return data

def get_dating():
    df = pd.read_csv('data/dating/ratings.dat', sep=',', error_bad_lines=False, encoding="latin-1", header=None)
    df.columns = ['user', 'item', 'rating']
    print('shape', df.shape)

    reader = Reader(rating_scale=(0, 10))
    data = Dataset.load_from_df(df.sample(n=100000, random_state=7), reader=reader)
    return data

def get_recipes():
    df = pd.read_csv('data/recipes/RAW_interactions.csv', sep=',', error_bad_lines=False, encoding="latin-1", header=None)
    df.drop(0, inplace=True)
    df.drop(4, axis='columns', inplace=True)
    df.drop(2, axis='columns', inplace=True)
    df.columns = ['user', 'item', 'rating']
    print('shape', df.shape)

    reader = Reader(rating_scale=(0, 10))
    data = Dataset.load_from_df(df.sample(n=100000, random_state=7), reader=reader)
    return data

def get_amazon_reviews_all():
    csv_url = 'http://snap.stanford.edu/data/amazon/productGraph/item_dedup.csv'

    df = pd.read_csv(csv_url, sep=',', error_bad_lines=False, encoding="latin-1", header=None)
    df.drop(3, axis='columns', inplace=True)
    df = df.sample(n=100000, random_state=7)
    df = df.reset_index(drop=True)
    df[0], _ = pd.factorize(df[0])
    df[1], _ = pd.factorize(df[1])

    reader = Reader(rating_scale=(0, 5))
    data = Dataset.load_from_df(df.sample(n=100000, random_state=7), reader=reader)
    return data

def get_amazon_reviews_movies():
    csv_url = 'http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Movies_and_TV.csv'

    df = pd.read_csv(csv_url, sep=',', error_bad_lines=False, encoding="latin-1", header=None)
    df.drop(3, axis='columns', inplace=True)
    df = df.sample(n=100000, random_state=7)
    df = df.reset_index(drop=True)
    df[0], _ = pd.factorize(df[0])
    df[1], _ = pd.factorize(df[1])

    reader = Reader(rating_scale=(0, 5))
    data = Dataset.load_from_df(df.sample(n=100000, random_state=7), reader=reader)
    return data

def get_amazon_reviews_books():
    csv_url = 'http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Books.csv'

    df = pd.read_csv(csv_url, sep=',', error_bad_lines=False, encoding="latin-1", header=None)
    df.drop(3, axis='columns', inplace=True)
    df = df.sample(n=100000, random_state=7)
    df = df.reset_index(drop=True)
    df[0], _ = pd.factorize(df[0])
    df[1], _ = pd.factorize(df[1])

    reader = Reader(rating_scale=(0, 5))
    data = Dataset.load_from_df(df.sample(n=100000, random_state=7), reader=reader)
    return data

def get_twitter_movie_ratings():
    csv_url = 'https://raw.githubusercontent.com/sidooms/MovieTweetings/master/snapshots/200K/ratings.dat'
    df = pd.read_csv(csv_url, sep='::', error_bad_lines=False, encoding="latin-1", header=None)
    df.drop(3, axis='columns', inplace=True)
    df.columns = ['user', 'item', 'rating']

    reader = Reader(rating_scale=(0, 10))
    data = Dataset.load_from_df(df.sample(n=100000, random_state=7), reader=reader)
    return data

def get_custom_dataset(dataset_name: str):
    if dataset_name == 'book-crossing':
        return get_book_crossing()
    elif dataset_name == 'dating':
        return get_dating()
    elif dataset_name == 'recipes':
        return get_recipes()
    elif dataset_name == 'amazon-reviews-all':
        return get_amazon_reviews_all()
    elif dataset_name == 'amazon-reviews-movies':
        return get_amazon_reviews_movies()
    elif dataset_name == 'amazon-reviews-books':
        return get_amazon_reviews_books()
    elif dataset_name == 'twitter-movie-ratings':
        return get_twitter_movie_ratings()
    else:
        raise NotImplementedError(f'Datasets currently supported are: {INBUILT_DATASETS + CUSTOM_DATASETS}')

def get_data(dataset_name: str):
    if dataset_name in INBUILT_DATASETS:
        return Dataset.load_builtin(dataset_name)
    return get_custom_dataset(dataset_name)

