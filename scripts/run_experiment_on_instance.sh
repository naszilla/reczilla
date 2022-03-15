#! /bin/bash
set -e

###############################################################################################################
# This script runs a single experiment on a gcloud instance.
# This instance needs to have a (not-necessarily-updated) Reczilla codebase in /home/shared/reczilla
#
# NOTE: this script requires that two environment variables are defined
# (use `export <var>=value` to define these)
#
# ARGS: this is a string of arguments that is passed to Experiment_handler.run_experiment
#  the following args should always be included in ARGS:
#  - `--split-dir /home/shared/split`  <- the location of the split data, which is copied here by this script
#  - `--result-dir /home/shared  <- where the results dir will be written
#  - `--write-zip`   <- to write the zip
#  - `--experiment-name <name>`  <- name of the new results dir
#  - `--split-type <name>  <- name of the split type. used for writing results to the correct dir
#  - `--alg-name <name>`  <- used for selecting the algorithm and writing results to the correct dir
#  - `--dataset-name <name>`  <- name of the dataset. used for writing results to the correct dir
#
# SPLIT_PATH_ON_BUCKET: full path to the dataset on the gcloud bucket (should start with gs://reczilla...)
#
###############################################################################################################

#############################################
# make sure environment variables are defined

if [ -n "$ARGS" ]; then
  echo "ARGS STRING: $ARGS"
else
  echo "ARGS string not defined" 1>&2
fi

if [ -n "$SPLIT_PATH_ON_BUCKET" ]; then
  echo "SPLIT_PATH_ON_BUCKET: $SPLIT_PATH_ON_BUCKET"
else
  echo "SPLIT_PATH_ON_BUCKET not defined" 1>&2
fi

###############
# prepare conda

source /home/shared/miniconda3/bin/activate
conda init
conda activate reczilla


#################
# copy split data

# copy all files in the split directory to a local folder
# first remove the split dir if it exists
rm -rf /home/shared/split
mkdir /home/shared/split

# copy the split data to /home/shared/split
gsutil cp "${SPLIT_PATH_ON_BUCKET}/*" /home/shared/split/

################
# run experiment

# run the experiment using command line args stored in variable $ARGS
# NOTE: the results should always be zipped and written to /home/shared/result.zip.
# NOTE: do this by passing positional argument result_dir = /home/shared
cd /home/shared/reczilla/RecSys2019_DeepLearning_Evaluation
python -m Experiment_handler.run_experiment ${ARGS}

# add a timestamp and a random string to the end of the filename, to avoid collisions
result_file=result_$(date +"%m%d%y_%H%M%S")_$(openssl rand -hex 2).zip
mv /home/shared/result.zip /home/shared/${result_file}

###############################
# save results to gcloud bucket

gsutil cp /home/shared/${result_file} gs://reczilla-results/inbox
