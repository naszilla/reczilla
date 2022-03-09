#! /bin/bash
# functions for running batch jobs
# load these functions by running 'source utils.sh'

function run_experiment() {

  # $1 = argument string passed to Experiment_handler.run_experiment
  # $2 = full path to the split data on the gcloud bucket (should start with gc://reczilla...)
  # $3 = instance name
  args_str=$1
  instance_name=$2
  split_path=$3

  # constants
  source_image=reczilla-v5-image
  service_account=default-compute-instance@research-collab-naszilla.iam.gserviceaccount.com
  zone=us-central1-a
  project=research-collab-naszilla

  echo "launching instance ${instance_name}..."
    gcloud beta compute instances create $instance_name --zone=$zone \
    --project=$project --image=$source_image \
    --service-account $service_account \
    --scopes=https://www.googleapis.com/auth/devstorage.read_write
  ret_code=$?
  echo "RETURN CODE from launching instance: $ret_code"

  # ssh and run the experiment. steps:
  # 1. pull the latest repo
  # 2. set environment variables used by script run_experiment_on_instance.sh
  # 3. chmod the experiment script
  # 4. run the experiment script
  instance_repo_dir=/home/shared/reczilla
  instance_script_location=${instance_repo_dir}/scripts/run_experiment_on_instance.sh

  echo "running experiment..."
  gcloud compute ssh --ssh-flag="-A" ${instance_name} --zone=${zone} --project=${project} \
  --command="\
  cd ${instance_repo_dir}; \
  git pull; \
  export ARGS=${args_str}; \
  export SPLIT_PATH_ON_BUCKET=${split_path}; \
  chmod +x ${instance_script_location}; \
  /bin/bash ${instance_script_location}"

  ret_code=$?
  echo "RETURN CODE from running experiment: $ret_code"

  echo "finished experiment. deleting instance..."
  gcloud compute instances delete ${instance_name}
  ret_code=$?
  echo "RETURN CODE from deleting instance: $ret_code"
}