
# Generating Reczilla "Meta"-data

The "meta"-data for Reczilla consists of the following for each datum:
- dataset parameters: multiple numerical metrics
- algorithm name (categorical)
- algorithm parameters: multiple numerical metrics
- algorithm performance: multiple numerical metrics

To generate this metadata, we need to train & test several different recommender algorithms on several different datasets (and splits). We generate sets of results by running **"experiments"**, each of which trains/tests a different dataset-split + algorithm combination.

This invoves several steps, outlined below.

## 1. Prepare Datasets

We first download all datasets to a local directory. This is done by `Data_manager.download_check_all_data`. This can be done in a single line:

```commandline
python -m Data_manager.download_check_all_data --data-dir /home/data/
```

Which will download all datasets, and write them in a standard format to local directory `/home/data`. Each experiment will create its own train/test split of each dataset, so we won't worry about that yet. 

**[INTERNAL ONLY]** We already ran this script on the gcloud instances `reczilla-vX`, and the data dir is `/home/shared/data`

## 2. Prepare Config Files

Each experiment is run using a config file, which looks like:

```commandline
# lines beginning with '#' are comments'
--dataset-name MyData
--num-samples 10
...
--arg arg_value
```

Where the first part of each line specifies an argument name, and the second part defines the argument value. (For context: these lines are parsed into a list of strings, and passed to an `argparse.ArgumentParser` object in `Experiment_handler.run_experiment`.)

Because these files are small, we generate one config file for every possible experiment we could run, using `Experiment_handler.generate_config_files`. With this script we can also specify other parameters of the experiment. This script will generate a directory structure of config files, that looks like this:

```commandline
config_directory/
├── Dataset_One/
│   ├── Split_One/
│       ├── Algorithm_A/config.txt
│       ...
│       └── Algorithm_Z/config.txt
│   ├── Split_Two/
│       ├── Algorithm_A/config.txt
│       ...
│       └── Algorithm_Z/config.txt
│   ...
│   └── Split_N/
├── Dataset_Two/
│   ├── Split_One/
│       ├── Algorithm_A/config.txt
│       ...
│       └── Algorithm_Z/config.txt
│   ...
│   └── Split_N/
...
└── Dataset_M/
```

That is, the config file for each combination of (dataset)-(split)-(algorithm) is found in file:

```commandline
config_directory/<dataset>/<split>/<algorithm>/config.txt
```

## Running Experiments

To run an experiment, we use bash scripts that do the following:
1. create a new gcloud instance
2. prepare the instance (update code, initialize conda)
3. copy a single config file to the instance
4. copy a script to the instance, which will run an experiment (`scripts/run_experiment_on_instance.sh`)
5. run the script from step (4)
6. copy the results to gcloud storage
7. [TODO] delete the instance

All of this is handled by function `run_experiment` in `scripts/utils.sh`.

So, it takes only two lines to run a single experiment using config file `/home/config.txt` from the directory `scripts`:

```commandline
source utils.sh
run_experiment /home/config.txt experiment-name
```

## Running Batch Experiments

**TODO**