#!/bin/bash

# run two algorithms on all datasets. use 2 parameter samples

# load functions
source ../utils.sh

###################
# set trap
# this will sync the log files, and delete all instances

mkdir ${PWD}/logs
LOG_DIR=${PWD}/logs
trap "sync_logs ${LOG_DIR}; delete_instances instance_list" EXIT
instance_list=()

###################
# define parameters

# base name for the gcloud instances
instance_base=twoalg

# name of the expeirment (this will be the name of the top-level results folder)
experiment_base=twoalg

# maximum number of experiments (background processes) that can be running
MAX_PROCESSES=10

# params
alg_seed=0
num_samples=2
param_seed=1

# define the split type
split_type=DataSplitter_leave_k_out

# bucket where split data is read. we expect split data to be in bucket_base/<dataset name>/<split name>
bucket_base=gs://reczilla-results/dataset-splits/splits-v3

# set of algorithms
alg_list=(
ItemKNNCF_cosine
P3alphaRecommender
)

# set of datasets - updated for v3
dataset_list=(
Anime
BookCrossing
CiaoDVD
Dating
Epinions
FilmTrust
Frappe
GoogleLocalReviews
Gowalla
Jester2
LastFM
MarketBiasAmazon
MarketBiasModCloth
MovieTweetings
Movielens100K
Movielens10M
Movielens1M
Movielens20M
MovielensHetrec2011
NetflixPrize
Recipes
Wikilens
)

#################
# run experiments

# store all log files here. sync these with gcloud
mkdir ${PWD}/logs
LOG_DIR=${PWD}/logs

num_experiments=0
for i in ${!alg_list[@]};
do
  for j in ${!dataset_list[@]};
  do

    # order of args in arg_str is:
    #    "dataset_name",
    #    "split_type",
    #    "alg_name",
    #    "split_dir",
    #    "alg_seed",
    #    "param_seed",
    #    "num_samples",
    #    "result_dir",
    #    "experiment_name",

    # argument string that will be passed to Experiment_handler.run_experiment
    # NOTE: in the current version of the split directory, the dataset names do not have suffix "Reader"
    split_path_on_bucket=${bucket_base}/${dataset_list[j]}/${split_type}

    arg_str="\
    ${dataset_list[j]}Reader \
    ${split_type} \
    ${alg_list[i]} \
    /home/shared/split \
    ${alg_seed} \
    ${param_seed} \
    ${num_samples}
    /home/shared \
    ${experiment_base}-${i}-${j} \
    ${split_path_on_bucket}"

    instance_name=${instance_base}-${i}-${j}

    run_experiment "${arg_str}" ${split_path_on_bucket} ${instance_base}-${i}-${j} >> ${LOG_DIR}/log_${i}_${j}_$(date +"%m%d%y_%H%M%S").txt 2>&1 &
    num_experiments=$((num_experiments + 1))

    # add instance name to the instance list
    instance_list+=("${instance_name}")

    echo "launched instance ${instance_base}-${i}-${j}. (job number ${num_experiments})"
    sleep 1

    # if we have started MAX_PROCESSES experiments, wait for them to finish
    wait_until_processes_finish $MAX_PROCESSES

  done
done

echo "still waiting for processes to finish..."
wait
echo "done."
