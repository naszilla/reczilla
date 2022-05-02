# Overview of Datasets on Google Cloud

All splits live in `gs:/reczilla-results/dataset-splits/`. Each sub-directory here contains a different set of splits.

## `gs:/reczilla-results/dataset-splits/splits-v3`

All but one of the directories here contain a single split, created using a random leave-one-out split (one random interaction is reserved for validation, for each dataset). The one exception is the directory `reczilla-results/dataset-splits/splits-v3/AmazonReviewData`, which contains a different sub-directory for each of the amazon splits.

**Note:** all splits here are in the `DataSplitter_leave_k_out_random` folders (see `-v5` below).

## `gs:/reczilla-results/dataset-splits/splits-v5`

This directory contains one sub-directory for each dataset, including all amazon splits (unlike the `-v3` folder). Each dataset sub-dir contains two sub-dirs, each for a different split:
- `DataSplitter_leave_k_out_last`: leave-last-out: the last interaction from each user is placed in the test set, and all remaining interactions are used for training.
- `DataSplitter_leave_k_out_random`: leave-one-out: one randomly-selected interaction is placed in the test set for each user. **Note:** this is the same split present in `splits-v3`. 


## Ignore these sub-directories
- `gs:/reczilla-results/dataset-splits/splits-v1`
- `gs:/reczilla-results/dataset-splits/splits-v2`
- `gs:/reczilla-results/dataset-splits/splits-v4`
