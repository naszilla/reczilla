from Data_manager.BookCrossing.BookCrossingReader import *
from Data_manager.Jester2.Jester2Reader import *
from Data_manager.Dating.DatingReader import *
from Data_manager.Wikilens.WikilensReader import *
from Data_manager.MarketBiasAmazon.MarketBiasAmazonReader import *
from Data_manager.MarketBiasModCloth.MarketBiasModClothReader import *
from Data_manager.MovieTweetings.MovieTweetingsReader import *
from Data_manager.Recipes.RecipesReader import *
data_reader = RecipesReader()
data_reader.load_data()