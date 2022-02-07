#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 4 Feb 2022

@author: Duncan C McElfresh

code for iterating over each dataset


"""

# multiple amazon reviewdata readers
from Data_manager.AmazonReviewData.AmazonMoviesTVReader import AmazonMoviesTVReader
from Data_manager.AmazonReviewData.AmazonMusicReader import AmazonMusicReader

# multiple movielens readers
from Data_manager.Movielens.Movielens100KReader import Movielens100KReader
from Data_manager.Movielens.Movielens1MReader import Movielens1MReader
from Data_manager.Movielens.Movielens10MReader import Movielens10MReader
from Data_manager.Movielens.Movielens20MReader import Movielens20MReader
from Data_manager.Movielens.MovielensHetrec2011Reader import MovielensHetrec2011Reader

# everything else!
from Data_manager.BookCrossing.BookCrossingReader import BookCrossingReader
from Data_manager.Dating.DatingReader import DatingReader
from Data_manager.Epinions.EpinionsReader import EpinionsReader
from Data_manager.FilmTrust.FilmTrustReader import FilmTrustReader
from Data_manager.Frappe.FrappeReader import FrappeReader
from Data_manager.Gowalla.GowallaReader import GowallaReader
from Data_manager.Jester2.Jester2Reader import Jester2Reader
from Data_manager.MarketBiasAmazon.MarketBiasAmazonReader import MarketBiasAmazonReader
from Data_manager.MarketBiasModCloth.MarketBiasModClothReader import (
    MarketBiasModClothReader,
)
from Data_manager.MovieTweetings.MovieTweetingsReader import MovieTweetingsReader
from Data_manager.NetflixPrize.NetflixPrizeReader import NetflixPrizeReader
from Data_manager.Recipes.RecipesReader import RecipesReader
from Data_manager.Wikilens.WikilensReader import WikilensReader

DATASET_READER_LIST = [
    AmazonMoviesTVReader,
    AmazonMusicReader,
    Movielens100KReader,
    Movielens1MReader,
    Movielens10MReader,
    Movielens20MReader,
    MovielensHetrec2011Reader,
    BookCrossingReader,
    DatingReader,
    EpinionsReader,
    FilmTrustReader,
    FrappeReader,
    GowallaReader,
    Jester2Reader,
    MarketBiasAmazonReader,
    MarketBiasModClothReader,
    MovieTweetingsReader,
    NetflixPrizeReader,
    RecipesReader,
    WikilensReader,
]

DATASET_READER_NAME_LIST = [c.__name__ for c in DATASET_READER_LIST]

DATASET_DICT = {
    name: c for name, c in zip(DATASET_READER_NAME_LIST, DATASET_READER_LIST)
}

def dataset_handler(dataset_reader_name):
    """
    Returns:
        - dataset reader object
    """

    assert (
        dataset_reader_name in DATASET_READER_NAME_LIST
    ), f"dataset reader name not recognized: {dataset_reader_name}"

    return DATASET_DICT[dataset_reader_name]
