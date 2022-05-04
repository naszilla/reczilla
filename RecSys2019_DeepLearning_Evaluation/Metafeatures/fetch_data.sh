#!/bin/bash

# Destination folder
dest=../all_data
# bucket where split data is read. we expect split data to be in bucket_base/<dataset name>/<split name>
bucket_base=gs://reczilla-results/dataset-splits

mkdir $dest

for version in splits-v3 splits-v5
do
  mkdir $dest/$version
  gsutil cp -r $bucket_base/$version/* $dest/$version
done

#bucket_base=gs://reczilla-results/dataset-splits/splits-v3

