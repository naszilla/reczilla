#!/bin/bash
# run a set (c) of baseline algs with all datasets, excluding amazon splits

####################################
# BEGIN: user-defined parameters

# time limit = 10hrs (in seconds)
time_limit=36000

# base name for the gcloud instances
instance_base=algs-c

# name of the expeirment
experiment_base=full-experiment-c

# maximum number of experiments (background processes) that can be running
MAX_PROCESSES=10

# params
alg_seed=0
num_samples=100
param_seed=3

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

alg_array=(
MatrixFactorization_BPR_Cython
IALSRecommender
PureSVDRecommender
NMFRecommender
SLIM_BPR_Cython
)

# END: bookkeeping - modify at your own risk
################################################################


##################################################
# BEGIN: run experiments - modify at your own risk

num_experiments=1
count=1
for i in ${!alg_array[@]};
do
  for j in ${!dataset_array[@]};
  do
    # get random alg name
    alg_name=${alg_array[i]}
    dataset_name=${dataset_array[j]}
    echo "ALG = $alg_name"
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
    ${experiment_base}-${count} \
    ${split_path_on_bucket}"

    instance_name=${instance_base}-${count}

    LOG_FILE=${LOG_DIR}/log_${count}_$(date +"%m%d%y_%H%M%S").txt

    run_experiment "${arg_str}" ${split_path_on_bucket} ${instance_name} >> ${LOG_FILE} 2>&1 &

    # add instance name to the instance list
    instance_list+=("${instance_name}")

    echo "launched instance ${instance_name}. (job number ${num_experiments})"
    sleep 1

    num_experiments=$((num_experiments + 1))

    # if we have started MAX_PROCESSES experiments, wait for them to finish
    wait_until_processes_finish $MAX_PROCESSES

    count=$((count + 1))
  done
done

echo "still waiting for processes to finish..."
wait
echo "done."

# END: run experiments - modify at your own risk
##################################################
