#! /bin/bash
# functions for running batch jobs
# load these functions by running 'source utils.sh'

function test_gcp_commands() {

  # $1 = config file
  # $2 = instance name
  config_file=$1
  instance_name=$2

  # constant
  source_machine_image=reczilla-v5
  service_account=default-compute-instance@research-collab-naszilla.iam.gserviceaccount.com
  zone=us-central1-a
  project=research-collab-naszilla

  # important locations on the instance
  instance_config_location=/home/shared/config.txt
  instance_script_location=/home/shared/run_experiment.sh

  # launch an instance
#  echo "launching instance ${instance_name}"
#  gcloud beta compute instances create $instance_name --zone=$zone --project=$project \
#  --source-machine-image=$source_machine_image \
#  --service-account=$service_account --scopes=https://www.googleapis.com/auth/devstorage.read_write

#  # copy config file to a set location on the instance
#  gcloud compute scp $config_file $instance_name:$instance_config_location
#
#  # copy run script to a set location on the instance (this avoids permissions issues)
#  gcloud compute scp ./run_experiment.sh $instance_name:$instance_script_location

  # ssh and run the experiment
  echo "running experiment"
  gcloud compute ssh --ssh-flag="-A" $instance_name --zone=$zone --project=$project \
  --command="/bin/bash ${instance_script_location}"

  echo "finished experiment"
  # TODO: close instance
}