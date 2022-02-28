
# Using Google Cloud Computing Platform

## Creating a new Instance
1. Create a VM instance (Compute Engine > VM Instances > Create Instance)
2. Select machine configuration (CPU & memory)
3. Select boot disk. Default size is 10gb; if dealing with large datasets, select a larger disk. (Downloading the original versions of all reczilla datasets requires more than the default 10gb.)

**Note:** You can use the machine image "reczilla-v2" which already includes the data, github repo, and python environment for this project. (See below.)

## Machine Image: reczilla-v4

The directory `/home/shared` contains all datasets and code for the reczilla project. Everyone should have read/write/execute permissions on `/home/shared`. if not, you can change whatever you'd like with sudo. 

### Github ssh Credentials

In order to pull/push from/to the reczilla github repo, you'll need github credentials on the GCP instance. Here are two options, Option 1 is preferred.

#### Option 1: ssh agent forwarding

1. If you don't already have an ssh keypair that authenticates you with github, [create one and add it to your account](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent).

2. Test your ssh key with github:

```commandline
ssh -T git@github.com
```

If authentication is successful, you should see a message like:

```commandline
> Hi <username>! You've successfully authenticated, but GitHub does not
> provide shell access.
```

If authentication fails, [here are some troubleshooting tips](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/testing-your-ssh-connection).

3. When you ssh into a GCP instance, add the ssh flag "-A" to pass your ssh credentials, like this:
```
gcloud compute ssh --ssh-flag="-A" <instance name> --zone=<zone> --project=<project> 
```

#### Option 2: use root ssh key

All users should be able to authenticate with gituhb as reczilla-dev using a private key owned by user "root", in `/home/shared/.ssh`. The following block in `/etc/ssh/ssh_config` points to this key:

```commandline
Host github.com
        HostName github.com
        User git
        IdentityFile /home/shared/.ssh/id_ed25519 
```

**NOTE:** Since this ssh key is owned by root, you need to use `sudo` with all git commands that require authentication. 

### Code

The reczilla codebase is in `/home/shared/reczilla`. Github credentials are already set up, with ssh key in `/home/shared/.ssh`. Make sure to `git pull` when starting from a new image.

### Python

Miniconda and the reczilla python env was installed fresh into `/home/shared/miniconda3`. NOTE: conda must be initialized by each user. To initialize, run the following:

```source /home/shared/miniconda3/bin/activate```

and then

```conda init```

To activate the reczilla env:

```conda activate reczilla```


### /home/shared/data

Contains all downloaded data. To check all datasets, and download any new datasets, run the following from the `reczilla` environment

```
cd /home/shared/reczilla/RecSys2019_DeepLearning_Evaluation/```

python -m Data_manager.download_check_all_data --data-dir /home/shared/data/
```


## Resizing a disk

If the current disk is too small, resizing it is easy (https://cloud.google.com/sdk/gcloud/reference/compute/disks/resize).

```
gcloud compute disks resize <DISK NAME> --size=<NEW SIZE>
```

After running this command, you can simply restart the instance and GCP will automatically reallocate storage to the main partition. You can also use `growpart`, but I haven't had success using this with GCP.

## Launching instances and submitting jobs from CLI

Soon we will be running lots of experiments, so here is the start of automation for it. First, make sure you have gcloud installed on your computer. Also, make sure you have ssh keys set up as explained above (e.g., make sure `ssh -T git@github.com` works).

The following command will start a new instance using the `reczilla-v5` machine image and with read/write permissions to our GCP buckets.

```commandline
gcloud beta compute instances create test-1 --zone=us-central1-a \
--project=research-collab-naszilla --source-machine-image=reczilla-v5 \
--service-account default-compute-instance@research-collab-naszilla.iam.gserviceaccount.com \
--scopes=https://www.googleapis.com/auth/devstorage.read_write
```

To ssh into the image and copy a file to a bucket:

```commandline
gcloud compute ssh --ssh-flag="-A" test-1 --zone=us-central1-a --project=research-collab-naszilla

gsutil cp test.out gs://reczilla-results
```

Here is the start of a script we will use soon to help launch experiments
```bash
function test_gcp_commands() {
  instance_name=test-2
  source_machine_image=reczilla-v5
  service_account=default-compute-instance@research-collab-naszilla.iam.gserviceaccount.com
  zone=us-central1-a
  project=research-collab-naszilla

  # launch an instance
  gcloud beta compute instances create $instance_name --zone=$zone --project=$project \
  --source-machine-image=$source_machine_image \
  --service-account=$service_account --scopes=https://www.googleapis.com/auth/devstorage.read_write

  # ssh and perform a simple operation
  gcloud compute ssh --ssh-flag="-A" $instance_name --zone=$zone --project=$project \
  --command="cd /home/shared/reczilla/RecSys2019_DeepLearning_Evaluation; \
  cp test.py gcp-test-command-worked.out"
}
```
