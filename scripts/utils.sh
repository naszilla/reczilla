#! /bin/bash
# functions for running batch jobs
# load these functions by running 'source utils.sh'

function run_experiment() {

  # $1 = config file
  # $2 = instance name
  config_file=$1
  instance_name=$2

  # constant
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

  # copy config file to a set location on the instance
  instance_config_location=/home/shared/config.txt

  gcloud compute scp $config_file $instance_name:$instance_config_location
  ret_code=$?
  echo "RETURN CODE from scp config file: $ret_code"

  # ssh and run the experiment. steps:
  # 1. pull the latest repo
  # 2. chmod the experiment script
  # 3. run the experiment script
  instance_repo_dir=/home/shared/reczilla
  instance_script_location=${instance_repo_dir}/scripts/run_experiment_on_instance.sh

  echo "running experiment..."
  gcloud compute ssh --ssh-flag="-A" $instance_name --zone=$zone --project=$project \
  --command="\
  cd ${instance_repo_dir}; \
  git pull; \
  chmod +x ${instance_script_location}; \
  /bin/bash ${instance_script_location}"

  ret_code=$?
  echo "RETURN CODE from running experiment: $ret_code"

  echo "finished experiment. deleting instance..."
  gcloud compute instances delete ${instance_name}
  ret_code=$?
  echo "RETURN CODE from deleting instance: $ret_code"
}