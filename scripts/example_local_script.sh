#!/bin/bash

# load functions
source utils.sh

###################
# define parameters

# base name for the gcloud instances
instance_base=dcm

# params
alg_seed=0
num_samples=3
param_seed=0

# define the split type
split_type=DataSplitter_leave_k_out

# bucket where split data is read. we expect split data to be in bucket_base/<dataset name>/<split name>
bucket_base=gs://reczilla-results/dataset-splits/splits-v2

# set of algorithms
alg_list=(
#CoClustering
#DELF_EF_RecommenderWrapper
#DELF_MLP_RecommenderWrapper
#EASE_R_Recommender
#RP3betaRecommender
GlobalEffects
#SLIM_BPR_Cython
#IALSRecommender
#INeuRec_RecommenderWrapper
#ItemKNNCF_asymmetric
ItemKNNCF_cosine
#Mult_VAE_RecommenderWrapper
#P3alphaRecommender
#PureSVDRecommender
SlopeOne
TopPop
#UserKNNCF_dice
#UserKNNCF_jaccard
)

# set of datasets
dataset_list=(
Jester2Reader
)

#################
# run experiments

for i in ${!alg_list[@]};
do
  for j in ${!dataset_list[@]};
  do

    # order of args is:
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
    arg_str="\
    ${dataset_list[j]} \
    ${split_type} \
    ${alg_list[i]} \
    /home/shared/split \
    ${alg_seed} \
    ${param_seed} \
    ${num_samples}
    /home/shared \
    gcp-experiment-${i}-${j}"

    # NOTE: in the current version of the split directory, the dataset names do not have suffix "Reader"
    dataset_name=${dataset_list[j]}
    dataset_folder_name=${dataset_name%Reader}

    split_path_on_bucket=${bucket_base}/${dataset_folder_name}/${split_type}

    run_experiment "${arg_str}" ${split_path_on_bucket} ${instance_base}-${i}-${j} >> ./log_${i}_${j}_$(date +"%m%d%y_%H%M%S").txt 2>&1 &
    sleep 1
  done
done
