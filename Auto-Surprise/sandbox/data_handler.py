from surprise import Dataset, Reader
from urllib.request import urlopen
from zipfile import ZipFile
import pandas as pd
import os
import requests

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

def get_recipes():
    df = pd.read_csv('data/recipes/RAW_interactions.csv', sep=',', error_bad_lines=False, encoding="latin-1", header=None)
    df.drop(0, inplace=True)
    df.drop(4, axis='columns', inplace=True)
    df.drop(2, axis='columns', inplace=True)
    df.columns = ['user', 'item', 'rating']

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

def get_netflix():
    df = pd.read_csv('data/netflix/ratings.csv', sep=',', error_bad_lines=False, encoding="latin-1", header=None)
    df.columns = ['user', 'item', 'rating']

    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df.sample(n=100000, random_state=7), reader=reader)
    return data

def get_wikilens():
    df = pd.read_csv('data/wikilens/out.wikilens-ratings', sep='\t', error_bad_lines=False, encoding="latin-1", header=None)
    df.drop(3, axis='columns', inplace=True)
    df.columns = ['user', 'item', 'rating']

    reader = Reader(rating_scale=(0, 5))
    data = Dataset.load_from_df(df.sample(n=100000, random_state=7), reader=reader)
    return data

def get_twitch():  #TODO: need to change the scale of ratings 
    df = pd.read_csv('data/twitch/100k_a.csv', sep=',', error_bad_lines=False, encoding="latin-1", header=None)
    df.drop(2, axis='columns', inplace=True)
    df[2] = df[4] - df[3]
    df.drop([3, 4], axis='columns', inplace=True)
    df.columns = ['user', 'item', 'rating']

    reader = Reader(rating_scale=(1, 97))
    data = Dataset.load_from_df(df.sample(n=100000, random_state=7), reader=reader)
    return data

def get_steam():  #TODO: need to change the scale of ratings 
    df = pd.read_csv('data/steam/steam-200k.csv', sep=',', error_bad_lines=False, encoding="latin-1", header=None)
    df.drop([2, 4], axis='columns', inplace=True)
    df[1], _ = pd.factorize(df[1])
    df[2] = df.groupby([0, 1])[3].transform('sum')
    df.drop(3, axis='columns', inplace=True)
    df = df.drop_duplicates(subset=[0, 1])
    df = df.reset_index(drop=True)
    df.columns = ['user', 'item', 'rating']

    reader = Reader(rating_scale=(1, 97))
    data = Dataset.load_from_df(df.sample(n=100000, random_state=7), reader=reader)
    return data

def get_market_bias_amazon():
    df = pd.read_csv('https://raw.githubusercontent.com/MengtingWan/marketBias/master/data/df_electronics.csv', error_bad_lines=False, encoding="latin-1")
    df = df[['user_id', 'item_id', 'rating']]
    df.columns = ['user', 'item', 'rating']

    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df.sample(n=100000, random_state=7), reader=reader)
    return data

def get_market_bias_modcloth():
    df = pd.read_csv('https://raw.githubusercontent.com/MengtingWan/marketBias/master/data/df_modcloth.csv', error_bad_lines=False, encoding="latin-1")
    df = df[['user_id', 'item_id', 'rating']]
    df['user_id'], _ = pd.factorize(df['user_id'])
    df.columns = ['user', 'item', 'rating']

    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df.sample(n=100000, random_state=7), reader=reader)
    return data

def get_goodreads():
    df = pd.read_csv('data/goodreads/goodreads_interactions_sampled.csv', error_bad_lines=False, encoding="latin-1")
    df.columns = ['user', 'item', 'rating']

    reader = Reader(rating_scale=(0, 5))
    data = Dataset.load_from_df(df.sample(n=100000, random_state=7), reader=reader)
    return data

def get_renttherunway():  #TODO: need to check difference between rating and fit
    df = pd.read_csv('data/renttherunway/renttherunway_interactions.csv', error_bad_lines=False, encoding="latin-1")
    df.columns = ['user', 'item', 'rating']

    reader = Reader(rating_scale=(2, 10))
    data = Dataset.load_from_df(df.sample(n=100000, random_state=7), reader=reader)
    return data

def get_epinions():
    df = pd.read_csv('data/epinions/epinions_interactions.csv', error_bad_lines=False, encoding="latin-1")
    df['user'], _ = pd.factorize(df['user'])
    df['item'], _ = pd.factorize(df['item'])
    df.columns = ['user', 'item', 'rating']

    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df.sample(n=100000, random_state=7), reader=reader)
    return data

def get_ciao_dvd():
    df = pd.read_csv('data/ciao-dvd/user_movie_ratings_v1_0.dat', sep='\t', error_bad_lines=False, encoding="latin-1", header=None)
    df[2] = df[2].astype(int)
    df.columns = ['user', 'item', 'rating']

    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df.sample(n=100000, random_state=7), reader=reader)
    return data

INBUILT_DATASETS = ['ml-100k', 'ml-1m', 'jester']
CUSTOM_DATASET_FUNCTION_MAPS = {
    'book-crossing': get_book_crossing, 
    'dating': get_dating, 
    'recipes': get_recipes, 
    'amazon-reviews-all': get_amazon_reviews_all, 
    'amazon-reviews-movies': get_amazon_reviews_movies, 
    'amazon-reviews-books': get_amazon_reviews_books, 
    'twitter-movie-ratings': get_twitter_movie_ratings,
    'netflix-challenge': get_netflix,
    'wikilens': get_wikilens,
    'twitch': get_twitch,
    'steam': get_steam,
    'market-bias-amazon': get_market_bias_amazon,
    'market-bias-modcloth': get_market_bias_modcloth,
    'goodreads': get_goodreads,
    'renttherunway': get_renttherunway,
    'epinions': get_epinions,
    'ciao-dvd': get_ciao_dvd,
}
CUSTOM_DATASETS = list(CUSTOM_DATASET_FUNCTION_MAPS.keys())

def get_custom_dataset(dataset_name: str):
    if dataset_name in CUSTOM_DATASET_FUNCTION_MAPS:
        return CUSTOM_DATASET_FUNCTION_MAPS[dataset_name]()
    else:
        raise NotImplementedError(f'Datasets currently supported are: {INBUILT_DATASETS + CUSTOM_DATASETS}')

def get_data(dataset_name: str):
    if dataset_name in INBUILT_DATASETS:
        return Dataset.load_builtin(dataset_name)
    return get_custom_dataset(dataset_name)

