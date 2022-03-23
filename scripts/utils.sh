#! /bin/bash
# functions for running batch jobs
# load these functions by running 'source utils.sh'

# constants
image_family=reczilla
service_account=default-compute-instance@research-collab-naszilla.iam.gserviceaccount.com
zone=us-central1-a
project=research-collab-naszilla

wait_until_processes_finish() {
  # only takes one arg: the maximum number of processes that can be running
  # print a '.' every 60 iterations
  counter=0
  while [ `jobs -r | wc -l | tr -d " "` -gt $1 ]; do
    sleep 1
    counter=$((counter+1))
    if (($counter % 60 == 0))
    then
      echo -n "."     # no trailing newline
    fi
  done
  echo "no more than $1 jobs are running. moving on."
}

run_experiment() {

  # $1 = argument string passed to Experiment_handler.run_experiment
  # $2 = full path to the split data on the gcloud bucket (should start with gc://reczilla...)
  # $3 = instance name
  args_str="$1"
  split_path="$2"
  instance_name="$3"

  echo "run_experiment: args_str: ${args_str}"
  echo "run_experiment: split_path: ${split_path}"
  echo "run_experiment: instance_name: ${instance_name}"

  # set a return trap to delete the instance when this function returns
  trap "echo deleting instance ${instance_name}...; gcloud compute instances delete ${instance_name} --zone=${zone} --project=${project}" RETURN

  # maximum number of attempts at creating gcloud instance and ssh
  MAX_TRIES=5

  echo "launching instance ${instance_name}..."

  COUNT=0
  while [ $COUNT -lt $MAX_TRIES ]; do

    # attempt to create instance
    gcloud compute instances create $instance_name --zone=$zone \
    --project=$project --image-family=$image_family \
    --machine-type=n1-highmem-2 \
    --service-account $service_account \
    --scopes=https://www.googleapis.com/auth/devstorage.read_write

    # keep this for later
    INSTANCE_RETURN_CODE=$?

    if [ $INSTANCE_RETURN_CODE -ne 0 ]; then
      # failed to create instance
      let COUNT=COUNT+1
      echo "failed to create instance during attempt ${COUNT}... (exit code: ${INSTANCE_RETURN_CODE})"
      if [[ $COUNT -ge $(( $MAX_TRIES - 1 )) ]]; then
        echo "too many tries. giving up."
        exit 1
      fi
      echo "trying again in 5 seconds..."
      sleep 5
    else
      # success!
      break
    fi
  done
  echo "successfully created instance: ${instance_name}"



  # ssh and run the experiment. steps:
  # 1. set environment variables used by script run_experiment_on_instance.sh
  # 2. chmod the experiment script
  # 3. run the experiment script
  instance_repo_dir=/home/shared/reczilla
  instance_script_location=${instance_repo_dir}/scripts/run_experiment_on_instance.sh

  COUNT=0
  while [ $COUNT -lt $MAX_TRIES ]; do

    # attempt to run experiment
    gcloud compute ssh --ssh-flag="-A" ${instance_name} --zone=${zone} --project=${project} \
      --command="\
      export ARGS=\"${args_str}\"; \
      export SPLIT_PATH_ON_BUCKET=${split_path}; \
      chmod +x ${instance_script_location}; \
      /bin/bash ${instance_script_location}"

    SSH_RETURN_CODE=$?

    if [ $SSH_RETURN_CODE -ne 0 ]; then
      # failed to run experiment
      let COUNT=COUNT+1
      echo "failed to run experiment during attempt ${COUNT}... (exit code: ${SSH_RETURN_CODE})"
      if [[ $COUNT -ge $(( $MAX_TRIES - 1 )) ]]; then
        echo "too many tries. giving up and deleting instance."
        gcloud compute instances delete ${instance_name} --zone=${zone}
        exit 1
      fi
      echo "trying again in 5 seconds..."
      sleep 5
    else
      # success!
      break
    fi
  done
  echo "successfully ran experiment"

  # remember we don't need to delete the instance here, because we set a return trap
}
