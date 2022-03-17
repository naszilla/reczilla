
# Generating & Processing Reczilla "Meta"-data

The "meta"-data for Reczilla consists of the following for each datum:
- dataset parameters: multiple numerical metrics
- algorithm name (categorical)
- algorithm parameters: multiple numerical metrics
- algorithm performance: multiple numerical metrics

To generate this metadata, we need to train & test several different recommender algorithms on several different datasets (and splits). We generate sets of results by running **"experiments"**, each of which trains/tests a different dataset-split + algorithm combination.

This involves several steps, outlined below.

## 1. Prepare Datasets

We first download all datasets to a local directory. This is done by `Data_manager.download_check_all_data`. This can be done in a single line:

```commandline
python -m Data_manager.download_check_all_data --data-dir /home/data/
```

Which will download all datasets, and write them in a standard format to local directory `/home/data`. Each experiment will create its own train/test split of each dataset, so we won't worry about that yet. 

**[INTERNAL ONLY]** We already ran this script on the gcloud instances `reczilla-vX`, and the data dir is `/home/shared/data`

## Running Batch Experiments

There are three template scripts for running batch experiments. I'd recommend running these from a dedicated instance, since we keep an ssh command running while any jobs are running.

You should be able to run any of the following scripts from an instance with the `reczilla` repo on it:
- https://github.com/naszilla/reczilla/blob/main/scripts/run_all_experiments.sh: this script runs all combination of datasets + algorithms in the lists defined in the script
- https://github.com/naszilla/reczilla/blob/main/scripts/tests/two_alg_all_datasets.sh: this script runs only two algorithms on all datasets
- https://github.com/naszilla/reczilla/blob/main/scripts/tests/two_datasets_all_algs.sh: this script runs all algorithms on only two datasets.



## Running Single Experiments

To run a _single_ experiment, we use bash scripts that do the following:
1. create a new gcloud instance
2. prepare the instance (update code, initialize conda)
3. run a script on the instance using `gcloud compute ssh`, and pass arguments directly
4. copy the results to gcloud storage
5. delete the instance

To run a _set_ of experiments, we have another script that handles everything. In total there are three scripts.. Each is described below.

### `scripts/example_local_script.sh`

This script launches all of the experiments in sequence, so this script should probably live on your local machine, or on an instance that is just for managing other instances. An example script for this purpose is [`example_local_script.sh`](https://github.com/naszilla/reczilla/blob/main/scripts/example_local_script.sh). This script makes one call to the bash function `run_experiment` (in `scripts/utils.sh`) for each experiment you want to run. 

The three main variables defined in this script are:
1. the GCP location of the split datasets (a gcloud bucket path)
2. the argument string you want to pass to [`Experiment_handler.run_experiment`](https://github.com/naszilla/reczilla/blob/main/RecSys2019_DeepLearning_Evaluation/Experiment_handler/run_experiment.py). This script takes positional command line arguments only.
3. the name of each instance you want to create

### `scripts/utils.sh`

This script just defines the function `run_experiment`, and it should also live on your local machine. This function takes the three pieces of information described above and does the following:

1. launches an instance
2. runs the experiment on the instance
3. deletes the instance

It tries the first two steps multiple times, in case an error is thrown. "Running the experiment" (step 2) consists of a single command to `gcloud compute ssh`, where we define some environment variables and run the script `scripts/run_experiment_on_instance.sh`. We describe that script next.

### `scripts/run_experiment_on_instance.sh`

This script runs an experiment from an instance, using two environment variables. Both of these environment variables should be set beforehand (e.g. using the function `run_experiment()`). These variables are:
- `ARGS`: the argument string that will be passed to Experiment_handler.run_experiment
- `SPLIT_PATH_ON_BUCKET`: the gcloud path to the split data. (should start with `gc://reczilla...`)

This script does the following:
1. prepare conda
2. copy the split data from `SPLIT_PATH_ON_BUCKET` to a local folder
3. run `Experiment_handler.run_experiment` using the arguments in `ARGS`.
4. zip the results, and write them to the gcloud directory `gs://reczilla-results/inbox`


# Processing Results

We have a single python script that does the following:
- download all result files in the gcloud "inbox" folder
- put these in a nice human-readable file structure
- turn each into a CSV
- merge all results into a single, massive CSV

This script is here: [process_inbox.py](https://github.com/naszilla/reczilla/blob/main/RecSys2019_DeepLearning_Evaluation/reczilla_analysis/process_inbox.py). To run this script, `cd` into the repo folder `RecSys2019_DeepLearning_Evaluation`, and run:

```
python -m reczilla_analysis.process_inbox <result directory>
```

Where "result directory" is an empty directory. This script `os.system` to call gsutil, so you need to have gcloud cli tools installed. 

### CSV Result Format

The CSVs contain one row for each set of hyperparameters, and many, many columns:
- basic metadata columns (name of dataset, splitter, algorithm, timestamp, etc.)
- any exception that is raised during the experiment (and NaN otherwise). This is useful for debugging.
- **hyperparameter columns: (prefix: param\_")** each hyperparam used by any algorithm gets its own column. the name of these columns starts with "param\_"
- **metric columns: (prefix: metric\_)** each metric has its own column, for each cutoff tested, and for the test and validation set (if val is used). All of these have the prefix "metric\_". For example, the column for metric "ACCURACY" on the test set, for cutoff 5, has name `test_metric_ACCURACY_cut_5`.


