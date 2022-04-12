#!/bin/bash

# Destination folder
dest=../all_data

# bucket where split data is read. we expect split data to be in bucket_base/<dataset name>/<split name>
bucket_base=gs://reczilla-results/dataset-splits/splits-v3

mkdir $dest

gsutil cp -r $bucket_base/* $dest