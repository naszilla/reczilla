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
from Data_manager.YahooMovies.YahooMoviesReader import *
data_reader = YahooMoviesReader()
dataset = data_reader.load_data()
dataset.print_statistics()