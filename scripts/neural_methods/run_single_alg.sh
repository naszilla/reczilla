#!/bin/bash
# run a single alg on all datasets. two vars need to be defined:
# 1) basename = the name of the instances and experiments that will be created
#    > example: basename="dl-delfef"
# 2) alg_name: name of the alg class, passed to function algorithm_handler()
#    > example: alg_name=DELF_EF_RecommenderWrapper
#
# there are 7 algs implemented that should be run on GPUs:
# - INeuRec_RecommenderWrapper
# - UNeuRec_RecommenderWrapper
# - SpectralCF_RecommenderWrapper
# - DELF_MLP_RecommenderWrapper
# - DELF_EF_RecommenderWrapper
# - MFBPR_Wrapper
# - Mult_VAE_RecommenderWrapper

####################################
# BEGIN: user-defined parameters

# time limit = 10hrs (in seconds) for each job - a single train/test cycle
time_limit=36000

# maximum number of experiments (background processes) that can be running
MAX_PROCESSES=10

# params
alg_seed=0
num_samples=1
param_seed=3

# define the split type (leave-last-out = llo)
split_type=DataSplitter_leave_k_out_last

# bucket where split data is read. we expect split data to be in bucket_base/<dataset name>/<split name>
# we expect that the folders here do not have the suffix "Reader".
bucket_base=gs://reczilla-results/dataset-splits/splits-v5

# END: user-defined parameters
####################################


################################################################
# BEGIN: bookkeeping - modify at your own risk

# load functions and variables
source ../utils.sh
source ../neurips_experiments/v5_dataset_list.sh

# set trap to sync log files and delete all instances
mkdir ${PWD}/logs
LOG_DIR=${PWD}/logs
trap "sync_logs ${LOG_DIR}; delete_instances instance_list" EXIT
instance_list=()

# END: bookkeeping - modify at your own risk
################################################################


##################################################
# BEGIN: run experiments - modify at your own risk

echo "running experiments for ALG = $alg_name"

num_experiments=1
count=1
for j in ${!v5_dataset_list_small[@]};
do
  # get alg name and dataset name
  dataset_name=${v5_dataset_list_small[j]}
  echo "DATASET = $dataset_name"

  # argument string that will be passed to Experiment_handler.run_experiment
  # NOTE: this assumes that dataset names do not have suffix "Reader". but the reader objects do have this suffix
  split_path_on_bucket=${bucket_base}/${dataset_name}/${split_type}

  arg_str="\
  ${time_limit} \
  ${dataset_name}Reader \
  ${split_type} \
  ${alg_name} \
  /home/shared/split \
  ${alg_seed} \
  ${param_seed} \
  ${num_samples}
  /home/shared \
  ${basename}-${count} \
  ${split_path_on_bucket}"

  instance_name=${basename}-${count}

  LOG_FILE=${LOG_DIR}/log_${count}_$(date +"%m%d%y_%H%M%S").txt

  run_experiment_GPU "${arg_str}" ${split_path_on_bucket} ${instance_name} >> ${LOG_FILE} 2>&1 &

  # add instance name to the instance list
  instance_list+=("${instance_name}")

  echo "launched instance ${instance_name}. (job number ${num_experiments})"
  sleep 1

  num_experiments=$((num_experiments + 1))

  # if we have started MAX_PROCESSES experiments, wait for them to finish
  wait_until_processes_finish $MAX_PROCESSES

  count=$((count + 1))
done

echo "still waiting for processes to finish..."
wait
echo "done."

# END: run experiments - modify at your own risk
##################################################
