import os

import numpy as np
from scipy import sparse
from sklearn.linear_model import LinearRegression

from Data_manager.MovieTweetings.MovieTweetingsReader import *
from Data_manager.DataSplitter_global_timestamp import DataSplitter_global_timestamp

from KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from Conferences.WWW.MultiVAE_our_interface.MultiVAE_RecommenderWrapper import (
	Mult_VAE_RecommenderWrapper,
)

from LIME_utils.perturb import perturb_rated_items, perturb_all_items

RECOMMENDERS = ["item-knn", "mult-vae"]
PERTURB_MODES = ['all-items', 'rated-items']

np.random.seed(42)

data_reader = MovieTweetingsReader()
dataset = data_reader.load_data()
dataset.print_statistics()


save_folder = f"./tmp_DATA_{data_reader.DATASET_SUBFOLDER}"

# Create a training-validation-test split, for example by leave-1-out
# This splitter requires the DataReader object and the number of elements to holdout
dataSplitter = DataSplitter_global_timestamp(data_reader, k_out_percent=20, use_validation_set=True, force_new_split=False)

# The load_data function will split the data and save it in the desired folder.
# Once the split is saved, further calls to the load_data will load the splitted data ensuring you always use the same split
dataSplitter.load_data(save_folder_path=save_folder)

# We can access the three URMs with this function and the ICMs (if present in the data Reader)
URM_train, URM_validation, URM_test = dataSplitter.get_holdout_split()

ICM_dict = dataSplitter.get_loaded_ICM_dict()

num_users, num_items = URM_train.shape


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
		recommender.fit(epochs=1)
		recommender.save_model(trained_model_path)


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


########## Generate the explanation ##########

indices_sorted = np.argsort(-weights)[:5]

if perturb_mode == 'all-items':
	items_sorted = indices_sorted
else:
	items_sorted = np.array(rated_items)[indices_sorted].tolist()

print(f"user: {user_num}, item: {item_num}, \
	predicted rating: {scores[0, item_num]}")
print(f'Important items for this rating: {items_sorted[:5]}')
