#! /bin/bash

cd /home/shared/reczilla
git pull

# init conda and activate the reczilla environment
source /home/shared/miniconda3/bin/activate
conda init
conda activate reczilla

# run the experiment
cd /home/shared/reczilla/RecSys2019_DeepLearning_Evaluation
python -m Experiment_handler.run_experiment --data-dir  /home/shared/data/ --dataset-name Movielens100KReader --split-type leave_k_out --split-seed 1 --alg-seed 1 --alg-name UserKNNCF_cosine --num-samples 10 --result-dir /home/shared/ --experiment-name TEST_EXPERIMENT




