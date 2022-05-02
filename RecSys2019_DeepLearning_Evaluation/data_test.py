from Data_manager.BookCrossing.BookCrossingReader import *
from Data_manager.Jester2.Jester2Reader import *
from Data_manager.Dating.DatingReader import *
from Data_manager.Wikilens.WikilensReader import *
from Data_manager.MarketBiasAmazon.MarketBiasAmazonReader import *
from Data_manager.MarketBiasModCloth.MarketBiasModClothReader import *
from Data_manager.MovieTweetings.MovieTweetingsReader import *
from Data_manager.Recipes.RecipesReader import *
from Data_manager.Goodreads.GoodreadsReader import *
from Data_manager.Flixster.FlixsterReader import *
from Data_manager.CiaoDVD.CiaoDVDReader import *
from Data_manager.Anime.AnimeReader import *
from Data_manager.LastFM.LastFMReader import *
from Data_manager.GoogleLocalReviews.GoogleLocalReviewsReader import *
data_reader = CiaoDVDReader()
dataset = data_reader.load_data()

from Data_manager.DataSplitter_leave_k_out import DataSplitter_leave_k_out
from Data_manager.DataSplitter import DataSplitter

splitter = DataSplitter_leave_k_out(data_reader, k_out_value=1, folder=f"./test_load_save_{data_reader.DATASET_SUBFOLDER}/split", leave_random_out=False)

splitter.load_data()
