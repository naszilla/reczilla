#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 31 Jan 2022

@author: Duncan C McElfresh

NOTE: requires a newer version of scikit-optimize than the original codebase. Used v0.9 for development
"""

from ParameterTuning.SearchAbstractClass import SearchAbstractClass

import numpy as np

from Utils.reczilla_utils import set_deterministic


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
        sampler_args=None,
        param_seed=0,
        alg_seed=0,
        raise_exceptions=False,
        write_log_every=10,
        metadata_dict=None,
    ):
        """
        search for the best set of hyperparameters using multiple random draws from the hyperparameter space

        pass additional args to the sampler using sampler_args

        if metadata_dict is passed, add this to the metadata dict
        """

        if sampler_args is None:
            sampler_args = {}

        hyperparam_rs = np.random.RandomState(param_seed)

        # sample random hyperparam values
        hyperparam_samples = parameter_search_space.random_samples(
            n_samples,
            rs=hyperparam_rs,
            sampler_type=sampler_type,
            sampler_args=sampler_args,
        )

        resume_from_saved = False  # not implemented
        metric_to_optimize = "MAP"  # not important
        save_model = "no"  # we never want to save the model
        evaluate_on_test = "all"  # always evaluate on test dataset
        recommender_input_args_last_test = None  # not needed
        save_metadata = (
            True  # incrementally save metadata to file. this is how we record results
        )

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
            raise_exceptions,
            metadata_dict=metadata_dict
        )

        # generate n_cases random hyperparameter draws
        for i_sample, hyperparams in enumerate(hyperparam_samples):
            if i_sample % write_log_every == 0:
                self._write_log(
                    "{}: Starting parameter set {} of {}\n".format(
                        self.ALGORITHM_NAME, i_sample + 1, n_samples
                    )
                )
            set_deterministic(
                alg_seed
            )  # reinitialize random states using the algorithm seed
            self._objective_function(hyperparams)

        self._write_log(
            "{}: Search complete. Output written to: {}\n".format(
                self.ALGORITHM_NAME, self.output_folder_path,
            )
        )
