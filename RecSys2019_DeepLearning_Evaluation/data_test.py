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
data_reader = CiaoDVDReader()
dataset = data_reader.load_data()
dataset.print_statistics()

##### uncomment if want to test e2e pipeline

# from Data_manager.DataSplitter_global_timestamp import DataSplitter_global_timestamp

# save_folder = f"./tmp_DATA_{data_reader.DATASET_SUBFOLDER}"

# # Create a training-validation-test split, for example by leave-1-out
# # This splitter requires the DataReader object and the number of elements to holdout
# dataSplitter = DataSplitter_global_timestamp(data_reader, k_out_percent=10, use_validation_set=True, force_new_split=True)

# # The load_data function will split the data and save it in the desired folder.
# # Once the split is saved, further calls to the load_data will load the splitted data ensuring you always use the same split
# dataSplitter.load_data(save_folder_path=save_folder)

# # We can access the three URMs with this function and the ICMs (if present in the data Reader)
# URM_train, URM_validation, URM_test = dataSplitter.get_holdout_split()

# ICM_dict = dataSplitter.get_loaded_ICM_dict()


# from Base.Evaluation.Evaluator import EvaluatorHoldout
# from Base.NonPersonalizedRecommender import TopPop


# cutoffs = [5, 10]
# evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=cutoffs)


# # We now fit and evaluate a non personalized algorithm
# recommender = TopPop(URM_train)
# recommender.fit()

# results_dict, results_run_string = evaluator_test.evaluateRecommender(recommender)
# print(f"Result of {recommender.RECOMMENDER_NAME} is:\n" + results_run_string)