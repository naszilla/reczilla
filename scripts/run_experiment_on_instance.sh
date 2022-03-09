#! /bin/bash
set -e
# init conda and activate the reczilla environment
source /home/shared/miniconda3/bin/activate
conda init
conda activate reczilla

###################
# check config file

# we need to have a config file in /home/shared/config.txt
config_file=/home/shared/config.txt
if [ -f "$config_file" ]; then
    echo "config file exists."
else
    echo "ERROR: config file does not exist."
fi

# read dataset name and split name from config file
dataset_name=$(grep "dataset-name" $config_file | cut -d' ' -f2)
split_name=$(grep "split-type" $config_file | cut -d' ' -f2)

echo "dataset: ${dataset_name}"
echo "split: ${split_name}"

#################
# copy split data

# location of split data on the bucket
split_data_bucket=gs://reczilla-results/dataset-splits/splits-v2

# copy all files in the split directory to a local folder
mkdir /home/shared/split
gsutil cp "${split_data_bucket}/${dataset_name}/${split_name}/*" /home/shared/split/


################
# run experiment

# run the experiment from /home/shared/config.txt
# the results should always be zipped and written to /home/shared/result.zip
# this is done using argument --write-zip /home/shared/result.zip (in the config file)
cd /home/shared/reczilla/RecSys2019_DeepLearning_Evaluation
python -m Experiment_handler.run_experiment config --config-file ${config_file}

# add a timestamp to the result file. add a random string to the end of the filename, to avoid collisions
result_file=result_$(date +"%m%d%y_%H%M%S")_$(openssl rand -hex 2).zip

mv /home/shared/results.zip /home/shared/${result_file}

# save results to gcloud bucket
gsutil cp /home/shared/${result_file} gs://reczilla-results/inbox



