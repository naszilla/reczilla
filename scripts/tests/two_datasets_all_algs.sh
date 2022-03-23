#!/bin/bash

# run two algorithms on all datasets. use 2 parameter samples

# load functions
source ../utils.sh

###################
# define parameters

# base name for the gcloud instances
instance_base=twodataset

# name of the expeirment (this will be the name of the top-level results folder)
experiment_base=twodataset

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
ItemKNNCF_asymmetric
ItemKNNCF_tversky
ItemKNNCF_euclidean
ItemKNNCF_cosine
ItemKNNCF_jaccard
ItemKNNCF_dice
UserKNNCF_asymmetric
UserKNNCF_tversky
UserKNNCF_euclidean
UserKNNCF_cosine
UserKNNCF_jaccard
UserKNNCF_dice
TopPop
GlobalEffects
Random
P3alphaRecommender
RP3betaRecommender
MatrixFactorization_FunkSVD_Cython
MatrixFactorization_AsySVD_Cython
MatrixFactorization_BPR_Cython
IALSRecommender
PureSVDRecommender
NMFRecommender
SLIM_BPR_Cython
SLIMElasticNetRecommender
EASE_R_Recommender
#INeuRec_RecommenderWrapper      # REMOVING SOME METHODS FOR NOW
#UNeuRec_RecommenderWrapper      # REMOVING SOME METHODS FOR NOW
#SpectralCF_RecommenderWrapper   # REMOVING SOME METHODS FOR NOW
Mult_VAE_RecommenderWrapper
#DELF_MLP_RecommenderWrapper     # REMOVING SOME METHODS FOR NOW
DELF_EF_RecommenderWrapper
#MFBPR_Wrapper                   # REMOVING SOME METHODS FOR NOW
CoClustering
SlopeOne
)

# set of datasets
dataset_list=(
Dating
Movielens100K
)

#################
# run experiments

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
    arg_str="\
    ${dataset_list[j]}Reader \
    ${split_type} \
    ${alg_list[i]} \
    /home/shared/split \
    ${alg_seed} \
    ${param_seed} \
    ${num_samples}
    /home/shared \
    ${experiment_base}-${i}-${j}"

    split_path_on_bucket=${bucket_base}/${dataset_list[j]}/${split_type}

    run_experiment "${arg_str}" ${split_path_on_bucket} ${instance_base}-${i}-${j} >> ./log_${i}_${j}_$(date +"%m%d%y_%H%M%S").txt 2>&1 &
    num_experiments=$((num_experiments + 1))
    echo "launched instance ${instance_base}-${i}-${j}. (job number ${num_experiments})"
    sleep 1

    # if we have started MAX_PROCESSES experiments, wait for them to finish
    wait_until_processes_finish $MAX_PROCESSES
  done
done

echo "still waiting for processes to finish..."
wait
echo "done."
