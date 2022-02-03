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

from ParameterTuning.algorithm_handler import algorithm_handler

from skopt.space import Real, Integer, Categorical


# Use a dataReader to load the data into sparse matrices
data_reader = Movielens100KReader()
loaded_dataset = data_reader.load_data(save_folder_path="./tmp_DATA_MOVIELENS/")

# In the following way you can access the entire URM and the dictionary with all ICMs
URM_all = loaded_dataset.get_URM_all()

# Create a training-validation-test split, for example by leave-1-out
# This splitter requires the DataReader object and the number of elements to holdout
dataSplitter = DataSplitter_leave_k_out(data_reader)
# dataSplitter = DataSplitter_k_fold_random(data_reader)

dataSplitter.load_data(save_folder_path="./tmp_DATA_SPLIT_MOVIELENS/")  # save_folder_path= "result_experiments/usage_example/data/")

# We can access the three URMs with this function and the ICMs (if present in the data Reader)
URM_train, URM_validation, URM_test = dataSplitter.get_holdout_split()

# Now that we have the split, we can create the evaluators.
# The constructor of the evaluator allows you to specify the evaluation conditions (data, recommendation list length,
# excluding already seen items). Whenever you want to evaluate a model, use the evaluateRecommender function of the evaluator object
evaluator_validation = EvaluatorHoldout(
    URM_validation, cutoff_list=[5], exclude_seen=False
)
evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[5, 10, 20], exclude_seen=False)

# output folders
output_folder = "./test/"

# iterate over multiple algorithms using the algorithm handler
alg_list = ["ItemKNNCF_jaccard", "P3alphaRecommender", "RP3betaRecommender", "SLIM_BPR_Cython"]


for alg_name in alg_list:
    # name of output file
    output_file_name_root = alg_name + "_randomsearch"

    # get a recommender class, hyperparameter search space, and search_input_recommender_args from the algorithm handler
    alg, parameter_search_space, search_input_recommender_args = algorithm_handler("ItemKNNCF_jaccard")

    # add the training dataset to recommender_input_args (this is then passed to the alg constructor...)
    search_input_recommender_args.CONSTRUCTOR_POSITIONAL_ARGS = [URM_train]

    # create a search object for the random parameter search
    # we need to re-initialize this for each algorithm
    parameterSearch = RandomSearch(
        alg,
        evaluator_validation=evaluator_validation,
        evaluator_test=evaluator_test,
    )

    # run a random parameter search
    parameterSearch.search(
        search_input_recommender_args,
        parameter_search_space,
        n_samples=3,
        output_folder_path=output_folder,
        output_file_name_root=output_file_name_root,
        sampler_type="Sobol",
        sampler_args={},
        sample_seed=0,
    )
