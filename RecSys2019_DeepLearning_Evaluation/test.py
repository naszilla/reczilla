from Data_manager.Movielens.Movielens100KReader import Movielens100KReader
from Data_manager.DataSplitter_leave_k_out import DataSplitter_leave_k_out
from Data_manager.DataSplitter_k_fold_random import DataSplitter_k_fold_random
from Base.Evaluation.Evaluator import EvaluatorHoldout

from Base.NonPersonalizedRecommender import TopPop
from GraphBased.P3alphaRecommender import P3alphaRecommender

from KNN.UserKNNCFRecommender import UserKNNCFRecommender
from KNN.ItemKNNCFRecommender import ItemKNNCFRecommender

from ParameterTuning.SearchAbstractClass import SearchInputRecommenderArgs
from ParameterTuning.RandomSearch import RandomSearch

from skopt.space import Real, Integer, Categorical


# Use a dataReader to load the data into sparse matrices
data_reader = Movielens100KReader()
loaded_dataset = data_reader.load_data()

# In the following way you can access the entire URM and the dictionary with all ICMs
URM_all = loaded_dataset.get_URM_all()

# Create a training-validation-test split, for example by leave-1-out
# This splitter requires the DataReader object and the number of elements to holdout
dataSplitter = DataSplitter_leave_k_out(data_reader)
# dataSplitter = DataSplitter_k_fold_random(data_reader)

dataSplitter.load_data()  # save_folder_path= "result_experiments/usage_example/data/")

# We can access the three URMs with this function and the ICMs (if present in the data Reader)
URM_train, URM_validation, URM_test = dataSplitter.get_holdout_split()

# Now that we have the split, we can create the evaluators.
# The constructor of the evaluator allows you to specify the evaluation conditions (data, recommendation list length,
# excluding already seen items). Whenever you want to evaluate a model, use the evaluateRecommender function of the evaluator object
evaluator_validation = EvaluatorHoldout(
    URM_validation, cutoff_list=[5], exclude_seen=False
)
evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[5, 10, 20], exclude_seen=False)

# define a recommender class and hyperparameter space
recommender_class = UserKNNCFRecommender

# the search space is just a dictionary of hyperparameters and their ranges
# ranges must be specified as skopt objects (Real, Integer, Categorical)
similarity_type = "tversky"
parameter_search_space = {
    "topK": Integer(5, 1000),
    "shrink": Integer(0, 1000),
    "similarity": Categorical([similarity_type]),
    "tversky_alpha": Real(low=0, high=2, prior="uniform"),
    "tversky_beta": Real(low=0, high=2, prior="uniform"),
    "normalize": Categorical([True]),
}

# specify args that are passed to the recommendation alg. constructor and passed to the fit() function
# (can specify both positional and kwargs)
# since the training dataset is generally passed as a constructor,
recommender_input_args = SearchInputRecommenderArgs(
    CONSTRUCTOR_POSITIONAL_ARGS=[URM_train],
    CONSTRUCTOR_KEYWORD_ARGS={},
    FIT_POSITIONAL_ARGS=[],
    FIT_KEYWORD_ARGS={},
)

# create a search object for the random parameter search
parameterSearch = RandomSearch(
    recommender_class,
    evaluator_validation=evaluator_validation,
    evaluator_test=evaluator_test,
)


output_folder = "./tmp/"
output_file_name_root = "tmp"

parameterSearch.search(
    recommender_input_args,
    parameter_search_space,
    n_samples=3,
    output_folder_path=output_folder,
    output_file_name_root=output_file_name_root,
    sampler_type="Sobol",
    sampler_args={},
    sample_seed=0,
)
