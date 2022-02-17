"""
...
"""
import shutil
import unittest

import numpy as np
import scipy.sparse

from ParameterTuning.RandomSearch import RandomSearch
from algorithm_handler import algorithm_handler, ALGORITHM_NAME_LIST
from Base.Evaluation.Evaluator import EvaluatorHoldout


N_HYPERPARAM_SAMPLES = 2
N_USERS = 100
N_ITEMS = 50
EPOCHS = 5
DENSITY = 0.01

SEED = 0
TEST_FOLDER = "./temp_alg_test/"


class TestCollabAlgorithms(unittest.TestCase):
    """
    test collaborative algorithms (only using URM)

    NOTE: this test uses subtests to iterate over each dataset. it's a good idea to run this from command line, because
    the IDE might not use the correct configuration:
    > python -m tests.algorithm_test
    """

    rs = np.random.RandomState(0)

    URM_test = scipy.sparse.random(
        N_USERS,
        N_ITEMS,
        density=DENSITY,
        format="csr",
        dtype=np.float32,
        random_state=rs,
    )
    URM_train = scipy.sparse.random(
        N_USERS,
        N_ITEMS,
        density=DENSITY,
        format="csr",
        dtype=np.float32,
        random_state=rs,
    )
    URM_validation = scipy.sparse.random(
        N_USERS,
        N_ITEMS,
        density=DENSITY / 2.0,
        format="csr",
        dtype=np.float32,
        random_state=rs,
    )

    evaluator_validation = EvaluatorHoldout(
        URM_validation, cutoff_list=[5], exclude_seen=False, verbose=False
    )
    evaluator_test = EvaluatorHoldout(
        URM_test, cutoff_list=[5], exclude_seen=False, verbose=False
    )

    def test_algorithms(self):
        for i, alg_name in enumerate(ALGORITHM_NAME_LIST):
            with self.subTest(alg_name=alg_name):
                (
                    alg,
                    space,
                    search_input_recommender_args,
                    max_samples,
                ) = algorithm_handler(alg_name)

                # add the training dataset to recommender_input_args (this is then passed to the alg constructor...)
                search_input_recommender_args.CONSTRUCTOR_POSITIONAL_ARGS = [
                    TestCollabAlgorithms.URM_train
                ]

                # use a small number of epochs
                if "epochs" in search_input_recommender_args.FIT_KEYWORD_ARGS:
                    search_input_recommender_args.FIT_KEYWORD_ARGS["epochs"] = EPOCHS

                # create a search object for the random parameter search
                # we need to re-initialize this for each algorithm
                parameterSearch = RandomSearch(
                    alg,
                    evaluator_validation=TestCollabAlgorithms.evaluator_validation,
                    evaluator_test=TestCollabAlgorithms.evaluator_test,
                    verbose=False,
                )

                # run a random parameter search
                parameterSearch.search(
                    search_input_recommender_args,
                    space,
                    n_samples=min(N_HYPERPARAM_SAMPLES, max_samples),
                    output_folder_path=TEST_FOLDER,
                    output_file_name_root=alg_name,
                    sampler_type="Sobol",
                    sampler_args={},
                    sample_seed=SEED,
                    raise_exceptions=True,  # by default exceptions are caught and saved. we want to raise them
                )


if __name__ == "__main__":
    unittest.main()

    # remove temp folders
    shutil.rmtree(TEST_FOLDER, ignore_errors=True)
