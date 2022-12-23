
# Using Google Cloud Computing Platform with RecZilla

## Creating an Instance (VM) for RecZilla

You can use the following procedure to create a GCP instance with all of the code, python environments, and data needed to run RecZilla. (By default, GCP instances do not come with much pre-installed software.)

### `reczilla` image family

First, create an instance using the `reczilla` family of disk images (see all of the available disk images [here](https://console.cloud.google.com/compute/images?tab=images&authuser=3&project=research-collab-naszilla)). An image family is a simple way to version-control disk images: creating a new image in the family will "depreciate" older images. 

To update the `reczilla` image family, follow the instructions [here](reczilla/scripts/update_reczilla_image.sh).

Creating an instance from an image family will use the **latest** image in that family. The following command will create an image from the `reczilla` family. Make sure to change the image name and machine type (bash variables at the top) as needed:

```
machine_type=n2-standard-4
zone=us-central1-a
project=research-collab-naszilla
image_family=reczilla

instance_name=test-reczilla

# attempt to create instance
gcloud beta compute instances create ${instance_name} \
--zone=${zone} \
--machine-type=${machine_type} \
--project=$project \
--image-family=$image_family 
```

Once you ssh into this image, you should find the following RecZilla resources. These are all located in directory `/home/shared`. **Remember:** this is your instance, so feel free to edit/delete/create directories or files. This will not impact the image family.

RecZilla resources:
- conda environment `reczilla`, located in `/home/shared/miniconda3/`. You may need to run `conda init`.
- a local version of the RecZilla repo: `/home/shared/reczilla`

## Setting up SSH Config for GCP instances

Use this command to automatically set up ssh config entries for all gcp instances:

```gcloud compute config-ssh --project=research-collab-naszilla```

## Dataset Splits

We store all pre-split datasets in a GCP bucket. The location of each split has the format:

```
gs://reczilla-results/dataset-splits/splits-v5/{dataset_name}/{split_type}
```

(we used split_type=`DataSplitter_leave_k_out_last` in our experiments). The directory `splits-v5` is the latest set of predefined splits. 

To copy a dataset split locally, you want to use something like the following:

```gsutil cp "gs://reczilla-results/dataset-splits/splits-v5/AmazonCellPhonesAccessories/DataSplitter_leave_k_out_last/*" /home/shared/split/```

## Github ssh Credentials

In order to pull/push from/to the reczilla github repo, you'll need github credentials on the GCP instance. The easiest way to do this is with SSH agent forwarding:

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


## Resizing a disk

If the current disk is too small, resizing it is easy (https://cloud.google.com/sdk/gcloud/reference/compute/disks/resize).

```
gcloud compute disks resize <DISK NAME> --size=<NEW SIZE>
```

After running this command, you can simply restart the instance and GCP will automatically reallocate storage to the main partition. You can also use `growpart`, but I haven't had success using this with GCP.
