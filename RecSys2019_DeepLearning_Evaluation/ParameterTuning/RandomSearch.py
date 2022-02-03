#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 31 Jan 2022

@author: Duncan C McElfresh

NOTE: requires a newer version of scikit-optimize than the original codebase. Used v0.9 for development
"""

from ParameterTuning.SearchAbstractClass import SearchAbstractClass

from skopt.sampler import Sobol, Lhs, Halton, Hammersly, Grid
from skopt.space import Real, Integer, Categorical

import numpy as np

SAMPLER_DICT = {
    "Sobol": Sobol,
    "Lhs": Lhs,
    "Halton": Halton,
    "Hammersly": Hammersly,
    "Grid": Grid,
}


class RandomSearch(SearchAbstractClass):

    ALGORITHM_NAME = "RandomSearch"

    def __init__(
        self,
        recommender_class,
        evaluator_validation=None,
        evaluator_test=None,
        verbose=True,
    ):

        super(RandomSearch, self).__init__(
            recommender_class,
            evaluator_validation=evaluator_validation,
            evaluator_test=evaluator_test,
            verbose=verbose,
        )

    def _evaluate_on_validation(self, current_fit_parameters):

        if self.evaluator_validation is not None:

            return super(RandomSearch, self)._evaluate_on_validation(
                current_fit_parameters
            )

        else:
            recommender_instance, train_time = self._fit_model(current_fit_parameters)

            return (
                {self.metric_to_optimize: 0.0},
                "",
                recommender_instance,
                train_time,
                None,
            )

    def search(
        self,
        recommender_input_args,
        parameter_search_space,
        n_samples=1,
        output_folder_path=None,
        output_file_name_root=None,
        sampler_type="Sobol",
        sampler_args={},
        sample_seed=0,
    ):
        """
        search for the best set of hyperparameters using multiple random draws from the hyperparameter space

        pass additional args to the sampler using sampler_args
        """

        hyperparam_rs = np.random.RandomState(sample_seed)

        assert (
            sampler_type in SAMPLER_DICT
        ), f"sampler type {sampler_type} not recognized. sampler_type must be one of {list(SAMPLER_DICT.keys())}."
        sampler = SAMPLER_DICT[sampler_type](**sampler_args)

        # validate and create search space. this code is borrowed from SearchAbstractClass
        skopt_types = [Real, Integer, Categorical]

        hyperparam_names = []
        hyperparam_spaces = []
        for name, hyperparam in parameter_search_space.items():
            if any(isinstance(hyperparam, sko_type) for sko_type in skopt_types):
                hyperparam_names.append(name)
                hyperparam_spaces.append(hyperparam)
            else:
                raise ValueError(
                    "{}: Unexpected parameter type: {} - {}".format(
                        self.ALGORITHM_NAME, str(name), str(hyperparam)
                    )
                )

        # sample hyperparameter values
        hyperparam_sample_list = sampler.generate(
            hyperparam_spaces, n_samples, random_state=hyperparam_rs
        )

        # put each hyperparameter sample (a list) into its own dict
        hyperparam_samples = [
            {
                name: hyperparam_sample_list[j][i_param]
                for i_param, name in enumerate(hyperparam_names)
            }
            for j in range(n_samples)
        ]

        resume_from_saved = False  # not implemented
        metric_to_optimize = "MAP"  # not important
        save_model = "no"  # we never want to save the model
        evaluate_on_test = "all"  # always evaluate on test dataset
        recommender_input_args_last_test = None  # not needed
        save_metadata = True  # incrementally save metadata to file. may be useful, not necessary

        self._set_search_attributes(
            recommender_input_args,
            recommender_input_args_last_test,
            metric_to_optimize,
            output_folder_path,
            output_file_name_root,
            resume_from_saved,
            save_metadata,
            save_model,
            evaluate_on_test,
            n_samples,
        )

        # generate n_cases random hyperparameter draws
        for i_sample, hyperparams in enumerate(hyperparam_samples):
            self._objective_function(hyperparams)

        self._write_log(
            "{}: Search complete. Output written to: {}\n".format(
                self.ALGORITHM_NAME, self.output_folder_path,
            )
        )
