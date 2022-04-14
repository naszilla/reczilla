# download all logs locally and identify all jobs that failed and succeeded
import argparse
import os
import pandas as pd
from pathlib import Path


def run(args):

    base_path = Path(args.base_dir)
    inbox_path = base_path.joinpath("inbox_logs")

    # make sure that directories we will create do not exist, and that the base dir does exist
    assert base_path.exists(), f"base dir does not exist: {base_path}"

    inbox_path.mkdir(exist_ok=True)

    # pull all files
    os.system(f"gsutil -m rsync gs://reczilla-results/inbox/logs {inbox_path}")

    # gather results here
    filename_list = []
    complete_list = []
    instance_name_list = []
    dataset_name_list = []
    split_type_list = []
    alg_name_list = []
    split_dir_list = []
    alg_seed_list = []
    param_seed_list = []
    num_samples_list = []
    result_dir_list = []
    experiment_name_list = []
    split_path_list = []
    time_lim_list = []



    # for each result matching pattern inbox/*.zip, place it in the appropriate folder
    for result_file in inbox_path.glob("log*.txt"):
        num_args = 10  # temp
        try:
            # read log file
            with result_file.open() as f:
                lines = f.readlines()

            # find important lines and check whether job completed
            args_str = ""
            instance_name_line = ""
            complete = False
            for i, l in enumerate(lines):

                args_prefix = "run_experiment: args_str:"
                instance_name_prefix = "run_experiment: instance_name:"
                if l.startswith(args_prefix):
                    args_str = l[len(args_prefix):]
                    args = args_str.split()
                    if len(args) < 10:
                        # the args overflowed to the next line... get these too
                        more_args = lines[i+1].split()
                        args.extend(more_args)

                    if len(args) == 10:
                        num_args = 10
                        offset = 0

                    elif len(args) == 11:
                        num_args = 11
                        offset = 1
                    else:
                        raise Exception(f"there must be 10 or 11 args. {len(args)} args found: {args}")


                elif l.startswith(instance_name_prefix):
                    instance_name_line = l[len(instance_name_prefix):]
                elif "successfully ran experiment" in l:
                    complete = True

            # get information from lines
            # info from args string
            dataset_name_list.append(args[0 + offset])
            split_type_list.append(args[1 + offset])
            alg_name_list.append(args[2 + offset])
            split_dir_list.append(args[3 + offset])
            alg_seed_list.append(args[4 + offset])
            param_seed_list.append(args[5 + offset])
            num_samples_list.append(args[6 + offset])
            result_dir_list.append(args[7 + offset])
            experiment_name_list.append(args[8 + offset])
            split_path_list.append(args[9 + offset])

            if num_args == 11:
                time_lim_list.append(args[0])
            else:
                time_lim_list.append(None)

            filename_list.append(result_file)
            complete_list.append(complete)
            instance_name_list.append(instance_name_line.strip())

        except Exception as e:
            print(f"could not process file, skipping: {result_file}")
            print("exception:")
            print(e)

    logs_df = pd.DataFrame({
        "log_file_path": filename_list,
        "job_complete": complete_list,
        "instance_name": instance_name_list,
        "dataset_name": dataset_name_list,
        "split_type": split_type_list,
        "alg_name": alg_name_list,
        "split_dir": split_dir_list,
        "alg_seed": alg_seed_list,
        "param_seed": param_seed_list,
        "num_samples": num_samples_list,
        "result_dir": result_dir_list,
        "experiment_name": experiment_name_list,
        "split_path": split_path_list,
    })

    results_path = base_path.joinpath("result_logs.csv")
    if results_path.exists():
        print(f"WARNING: overwriting results file {results_path}")
    logs_df.to_csv(results_path, index=False, sep=";")

    print("done")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "base_dir",
        type=str,
        help="base directory where files will be downloaded and structured.",
    )

    args = parser.parse_args()

    run(args)
