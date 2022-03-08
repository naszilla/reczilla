#! /bin/bash

# init conda and activate the reczilla environment
source /home/shared/miniconda3/bin/activate
conda init
conda activate reczilla

# run the experiment from /home/shared/config.txt
# the results should always be zipped and written to /home/shared/result.zip
# this is done using argument --write-zip /home/shared/result.zip (in the config file)
cd /home/shared/reczilla/RecSys2019_DeepLearning_Evaluation
python -m Experiment_handler.run_experiment config --config-file /home/shared/config.txt

# add a timestamp to the result file. add a random string to the end of the filename, to avoid collisions
result_file=result_$(date +"%m%d%y_%H%M%S")_$(openssl rand -hex 2).zip

mv /home/shared/results.zip /home/shared/${result_file}

# save results to gcloud bucket
gsutil cp /home/shared/${result_file} gs://reczilla-results/inbox



