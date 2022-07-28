import numpy as np
from scipy import sparse

# NOTE: need to think about size of perturbation
# currenty it is Gaussain with std_dev=1

def perturb_all_items(user_history, num_perturbed_pts):
	num_items = user_history.shape[1]

	random_matrix = sparse.random(num_perturbed_pts, num_items, 
								density=0.01, data_rvs=np.random.randn)

	URM_perturbed = user_history[np.zeros(num_perturbed_pts), :] + random_matrix 
	
	return URM_perturbed

def perturb_rated_items(user_history, items_rated, num_perturbed_pts):
	num_items = user_history.shape[1]

	# gaussian noise -- sampled from randn: Normal(0,1)  
	random_data = np.random.randn(items_rated.size * num_perturbed_pts)

	# in CSR matrices, data[k] = a[row_ind[k], col_ind[k]] 
	row_ind = np.repeat(np.arange(num_perturbed_pts), items_rated.size) # [0,0,0,1,1,1,...]
	col_ind = np.tile(items_rated, num_perturbed_pts) # [1,2,4,1,2,4,...]

	random_matrix = sparse.csr_matrix((random_data, (row_ind, col_ind)), 
									shape=(num_perturbed_pts, num_items))

	URM_perturbed = user_history[np.zeros(num_perturbed_pts), :] + random_matrix 

	return URM_perturbed, random_data