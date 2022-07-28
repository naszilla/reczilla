import os
import numpy as np
from sklearn.linear_model import LinearRegression
from tabulate import tabulate

from Data_manager.Movielens.Movielens100KReader import Movielens100KReader
from Data_manager.AmazonReviewData.AmazonAllBeautyReader import AmazonAllBeautyReader 
from Data_manager.DataSplitter_global_timestamp import DataSplitter_global_timestamp

from KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from Conferences.WWW.MultiVAE_our_interface.MultiVAE_RecommenderWrapper import (
	Mult_VAE_RecommenderWrapper,
)
# from Conferences.KDD.CollaborativeDL_our_interface.CollaborativeDL_Matlab_RecommenderWrapper import (
# 	CollaborativeDL_Matlab_RecommenderWrapper,
# )
# from Conferences.KDD.CollaborativeVAE_our_interface.CollaborativeVAE_RecommenderWrapper import (
# 	CollaborativeVAE_RecommenderWrapper,
# )

from Base.Evaluation.Evaluator import EvaluatorHoldout


from LIME_utils.perturb import perturb_rated_items, perturb_all_items

RECOMMENDERS = ["item-knn", "mult-vae"]
PERTURB_MODES = ['all-items', 'rated-items']

np.random.seed(42)

data_reader = Movielens100KReader(reload_from_original_data='always')
dataset = data_reader.load_data()
dataset.print_statistics()


save_folder = f"./tmp_DATA_{data_reader.DATASET_SUBFOLDER}"

# Create a training-validation-test split, for example by leave-1-out
# This splitter requires the DataReader object and the number of elements to holdout
dataSplitter = DataSplitter_global_timestamp(data_reader, k_out_percent=20, 
				use_validation_set=True, force_new_split=False)

# The load_data function will split the data and save it in the desired 
# folder. Once the split is saved, further calls to the load_data will 
# load the splitted data ensuring you always use the same split
dataSplitter.load_data(save_folder_path=save_folder)

# Get the three URMs and the ICMs (if present in the data Reader)
URM_train, URM_validation, URM_test = dataSplitter.get_holdout_split()

# ICM_dict = {'ICM_metadata': CSC sparse matrix n_items*n_feats }
ICM_dict = dataSplitter.get_loaded_ICM_dict()
ICM = ICM_dict['ICM_genre'] if 'ICM_genre' in ICM_dict else None

n_users, n_items = URM_train.shape


########## Recsys model part ##########

recommender_name = "item-knn"

if recommender_name == "item-knn":
	recommender = ItemKNNCFRecommender(URM_train)
	recommender.fit()
	
elif recommender_name == "mult-vae":
	recommender = Mult_VAE_RecommenderWrapper(URM_train)

	trained_model_path = './Conferences/WWW/MultiVAE_our_interface/trained_model/'

	if os.path.exists(trained_model_path):
		recommender.load_model(trained_model_path)
	else:
		recommender.fit(epochs=500)
		recommender.save_model(trained_model_path)

elif recommender_name == 'item-knn-cbf':
	recommender = ItemKNNCBFRecommender(URM_train, ICM)
	recommender.fit()

# elif recommender_name == "collaborative-vae":
# 	recommender = CollaborativeVAE_RecommenderWrapper(URM_train, ICM)
# 	recommender.fit()

evaluator = EvaluatorHoldout(URM_test, cutoff_list=[5])
results_run, results_run_string = evaluator.evaluateRecommender(recommender)
print(results_run_string)

########## Generate the surrogate's dataset ##########


user_num = 3
user_history = URM_train[user_num]

scores = recommender._compute_item_score([user_num])
item_num = np.argmax(scores[0])

# in CSR matrices, column indices for row i are stored in indices[indptr[i]:indptr[i+1]]
# and their corresponding values are stored in data[indptr[i]:indptr[i+1]]
indptr = URM_train.indptr
start_ptr, end_ptr = indptr[user_num], indptr[user_num + 1] 
rated_items = URM_train.indices[start_ptr:end_ptr]
user_history_projected = URM_train.data[start_ptr:end_ptr]


num_perturbed_pts = 1000

perturb_mode = 'rated-items'

if perturb_mode == 'all-items':
	URM_perturbed =	perturb_all_items(user_history, num_perturbed_pts)

else:
	URM_perturbed, random_data = \
		perturb_rated_items(user_history, rated_items, num_perturbed_pts)

scores_perturbed = recommender._compute_item_score_for_new_URM(URM_perturbed)


########## Train the surrogate ##########

# NOTE: need to add regularization! ElasticNet is probably a good choice 

if perturb_mode == 'all-items':
	URM_surrogate = URM_perturbed.toarray()
else:
	# = URM_perturbed[:, items_rated] but column slicing is slow on CSR matrices
	URM_surrogate = np.tile(user_history_projected, (num_perturbed_pts, 1)) + \
					random_data.reshape(num_perturbed_pts, rated_items.size)

X = URM_surrogate
y = scores_perturbed[:, item_num]

surrogate = LinearRegression().fit(X, y)
weights = surrogate.coef_


########## Generate and pretty print the explanation ##########

item_name = data_reader.item_name


indices_sorted = np.argsort(-weights)

if perturb_mode == 'all-items':
	items_sorted = indices_sorted
else:
	items_sorted = np.array(rated_items)[indices_sorted].tolist()

top_3_items = items_sorted[:3]
bottom_3_items = items_sorted[-3:]

table_header = ['item index', 'item name', 
				'item rating', 'item predicted rating', 'weight']

def generate_table_row(item_index):
	return [item_index, item_name[item_index],
			URM_train[user_num, item_index], round(scores[0,item_index]), 
			round(weights[list(rated_items).index(item_index)], 4)]

print(f'User of interest: user_id: {user_num}')

print('Item under inspection:')
table = [table_header, generate_table_row(item_num)]
print(tabulate(table), end='\n')

print('Most relevant items:')
table = [table_header] + [generate_table_row(i) for i in top_3_items]
print(tabulate(table), end='\n')

print('Least relevant items:')
table = [table_header] + [generate_table_row(i) for i in bottom_3_items]
print(tabulate(table), end='\n')
