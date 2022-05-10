#!/bin/bash

dest=../metadatasets

mkdir $dest

gsutil cp -r gs://reczilla-results/meta-datasets/* $dest