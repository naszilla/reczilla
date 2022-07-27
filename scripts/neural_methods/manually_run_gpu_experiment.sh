
#################################
# define args here

bucket_base=gs://reczilla-results/dataset-splits/splits-v5

# name of the instance
instance_name=manual-gpu-test

# experiment args
time_limit=72000 # 20 hrs in seconds
dataset_name=Movielens1M
split_type=DataSplitter_leave_k_out_last
alg_name=INeuRec_RecommenderWrapper
alg_seed=0
num_samples=1
param_seed=3
experiment_name=manual-gpu-test
split_path_on_bucket=${bucket_base}/${dataset_name}/${split_type}

# put these all in a string
args_str="\
    ${time_limit} \
    ${dataset_name}Reader \
    ${split_type} \
    ${alg_name} \
    /home/shared/split \
    ${alg_seed} \
    ${param_seed} \
    ${num_samples}
    /home/shared \
    ${experiment_name} \
    ${split_path_on_bucket}"

#################################


# constants
image_family=reczilla
zone=us-central1-a
project=research-collab-naszilla
ACCELERATOR_TYPE=nvidia-tesla-t4
ACCELERATOR_COUNT=1

# create instance
gcloud compute instances create $instance_name --zone=$zone \
--project=$project --image-family=$image_family \
--machine-type=n1-highmem-2 \
--accelerator type=${ACCELERATOR_TYPE},count=${ACCELERATOR_COUNT} \
--maintenance-policy TERMINATE \
--scopes=https://www.googleapis.com/auth/devstorage.read_write

instance_repo_dir=/home/shared/reczilla
instance_script_location=${instance_repo_dir}/scripts/run_experiment_on_instance.sh

# attempt to run experiment (or, just ssh in and run the commands below)
gcloud compute ssh --ssh-flag="-A" ${instance_name} --zone=${zone} --project=${project} \
  --command="\
  export ARGS=\"${args_str}\"; \
  export SPLIT_PATH_ON_BUCKET=${split_path_on_bucket}; \
  chmod +x ${instance_script_location}; \
  /bin/bash ${instance_script_location}"
