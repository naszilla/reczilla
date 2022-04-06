#!/bin/bash
# run a random sample of dataset + algorithm pairs

####################################
# BEGIN: user-defined parameters

# number of algorithm-dataset pairs to sample
num_pairs=5

# base name for the gcloud instances
instance_base=random-pairs

# name of the expeirment
experiment_base=random-pairs

# maximum number of experiments (background processes) that can be running
MAX_PROCESSES=10

# params
alg_seed=0
num_samples=6
param_seed=0

# define the split type
split_type=DataSplitter_leave_k_out

# bucket where split data is read. we expect split data to be in bucket_base/<dataset name>/<split name>
# we expect that the folders here do not have the suffix "Reader".
bucket_base=gs://reczilla-results/dataset-splits/splits-v3

# END: user-defined parameters
####################################


################################################################
# BEGIN: bookkeeping - modify at your own risk

# load functions
source utils.sh

# set trap to sync log files and delete all instances
mkdir ${PWD}/logs
LOG_DIR=${PWD}/logs
trap "sync_logs ${LOG_DIR}; delete_instances instance_list" EXIT
instance_list=()

# END: bookkeeping - modify at your own risk
################################################################


##################################################
# BEGIN: run experiments - modify at your own risk

num_experiments=0
count=1
while [ $count -lt $num_pairs ]; do
  # get random alg name
  alg_name=$(random_alg)
  dataset_name=$(random_dataset)
  echo "---- PAIR $count ----"
  echo "ALG = $alg_name"
  echo "DATASET = $dataset_name"

  # argument string that will be passed to Experiment_handler.run_experiment
  # NOTE: this assumes that dataset names do not have suffix "Reader". but the reader objects do have this suffix
  split_path_on_bucket=${bucket_base}/${dataset_name}/${split_type}

  arg_str="\
  ${dataset_name}Reader \
  ${split_type} \
  ${alg_name} \
  /home/shared/split \
  ${alg_seed} \
  ${param_seed} \
  ${num_samples}
  /home/shared \
  ${experiment_base}-${count} \
  ${split_path_on_bucket}"

  instance_name=${instance_base}-${count}

  echo "run_experiment "${arg_str}" ${split_path_on_bucket} ${instance_name} >> ${LOG_DIR}/log_${count}_$(date +"%m%d%y_%H%M%S").txt 2>&1 &"
  num_experiments=$((num_experiments + 1))

  # add instance name to the instance list
  instance_list+=("${instance_name}")

  echo "launched instance ${instance_name}. (job number ${num_experiments})"
  sleep 1

  # if we have started MAX_PROCESSES experiments, wait for them to finish
  wait_until_processes_finish $MAX_PROCESSES

  count=$((count + 1))
done

echo "still waiting for processes to finish..."
wait
echo "done."

# END: run experiments - modify at your own risk
##################################################
