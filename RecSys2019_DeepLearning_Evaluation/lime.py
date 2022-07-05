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
RECOMMENDERS = ["item-knn", "mult-vae"]


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

recommender_name = "mult-vae"

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

user_num = 0
URM_row = URM_train[user_num]

scores = recommender._compute_item_score([user_num])

item_num = np.argmax(scores[0])


num_perturbed_pts = 1000

np.random.seed(42)
# gaussian noise -- sampled from randn: Normal(0,1)  
random_matrix = sparse.random(num_perturbed_pts, num_items, 
							density=0.01, data_rvs=np.random.randn)

URM_perturbed = URM_row[np.zeros(num_perturbed_pts), :] + random_matrix 


scores_perturbed = recommender._compute_item_score_for_new_URM(URM_perturbed)


########## Train the surrogate ##########

X = URM_perturbed.todense()
y = scores_perturbed[:, item_num]

surrogate = LinearRegression().fit(X, y)
weights = surrogate.coef_


########## Generate the explanation ##########

imp_items = np.argsort(-weights)[:5]

print(f"user: {user_num}, item: {item_num}, \
	predicted rating: {scores[0, item_num]}")
print(f'Important items for this rating: {imp_items}')
