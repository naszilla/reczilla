
# Generating Reczilla "Meta"-data

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

## Running Experiments

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
