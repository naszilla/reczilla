# train a basic model and save metrics

import numpy as np
import pandas as pd

from utils import generate_filepath, get_logger

import hyperopt.pyll.stochastic

import surprise.accuracy
from surprise.model_selection import train_test_split
from data_handler import get_data

from model_handler import ALL_ALGORITHMS, ALL_SPACES


def run():

    logger = get_logger(logfile=None)

    verbose = False
    num_samples = 3
    out_dir = "./metrics"
    dataset_list = ["dating", "book-crossing"]
    algorithm_list = ["constant", "baseline_only", "normal_predictor"]

    seed = 0
    test_size = 0.25

    # collect results in a dataframe
    columns = [
        "dataset",
        "model",
        "params",
        "param_sample_number",
        "metric_fcp",
        "metric_rmse",
        "metric_mse",
        "metric_mae",
    ]
    df = pd.DataFrame(columns=columns)

    for dataset_name in dataset_list:

        logger.info(f"dataset: {dataset_name}")

        # retrieve data
        data = get_data(dataset_name)

        # generate a train-test split
        split_rs = np.random.RandomState(seed)
        train, test = train_test_split(data, test_size=test_size, random_state=split_rs)

        # for each algorithm, train using sets of random parameters
        for alg_name in algorithm_list:
            logger.info(f"starting algorithm: {alg_name}")
            space = ALL_SPACES[alg_name]
            alg_handle = ALL_ALGORITHMS[alg_name]

            for i_sample in range(num_samples):

                # sample a parameter set
                params = hyperopt.pyll.stochastic.sample(space)
                print("params")
                print(params)

                # initialize algorithm
                if params is None:
                    alg = alg_handle()
                else:
                    alg = alg_handle(**params)

                print(alg)

                # train algorithm
                alg.fit(train)

                # calculate metrics for the model
                logger.info("calculating metrics...")

                y = alg.test(test)

                # write the model class and accuracy methods to file
                # TODO: other metrics to add:
                # TODO: - hit rate @ N (HR@N)
                # TODO: - rank-based metrics (NDCG, MRR, MAP, ...)
                row = {}

                row["dataset"] = dataset_name
                row["model"] = alg_name
                row["param_sample_number"] = i_sample
                row["params"] = str(params)
                row["metric_fcp"] = surprise.accuracy.fcp(y, verbose=verbose)
                row["metric_rmse"] = surprise.accuracy.rmse(y, verbose=verbose)
                row["metric_mse"] = surprise.accuracy.mse(y, verbose=verbose)
                row["metric_mae"] = surprise.accuracy.mae(y, verbose=verbose)

                df = df.append(row, ignore_index=True)

    # write results df to file
    out_file = generate_filepath(out_dir, "test", "csv")

    logger.info(f"writing results to file: {out_file}...")

    df.to_csv(out_file, index=False, sep=";")

    logger.info("done")


if __name__ == "__main__":
    run()
