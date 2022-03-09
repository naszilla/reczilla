#!/bin/bash

# load functions
source utils.sh
#conda activate reczilla

# generate config files locally
# python -m Experiment_handler.generate_config_files ...

config_dir=/Users/duncan/research/active_projects/reczilla/RecSys2019_DeepLearning_Evaluation/ONE_SAMPLE_EXPT

# run experiments

alg_list=(
CoClustering
#DELF_EF_RecommenderWrapper
#DELF_MLP_RecommenderWrapper
#EASE_R_Recommender
#RP3betaRecommender
#GlobalEffects
#SLIM_BPR_Cython
#IALSRecommender
#INeuRec_RecommenderWrapper
#ItemKNNCF_asymmetric
#ItemKNNCF_cosine
#Mult_VAE_RecommenderWrapper
#P3alphaRecommender
#PureSVDRecommender
#SlopeOne
#TopPop
#UserKNNCF_dice
#UserKNNCF_jaccard
)

for i in ${!alg_list[@]};
do
  run_experiment ${config_dir}/Jester2Reader/DataSplitter_leave_k_out/${alg_list[i]}/config.txt dcm-test-${i} > ./log_${i}.txt
  sleep 1
done

# pass