from skopt.sampler import Sobol, Lhs, Halton, Hammersly, Grid
from skopt.space import Real, Integer, Categorical

SAMPLER_DICT = {
    "Sobol": Sobol,
    "Lhs": Lhs,
    "Halton": Halton,
    "Hammersly": Hammersly,
    "Grid": Grid,
}


class ParameterSpace:
    """
    generic class for hyperparameter spaces.

    two variables to initialize:
    - param_dict: dictionary of variables and ranges
    - default: default param set
    """

    def __init__(self, param_dict, default):

        # validate and create search space. this code is borrowed from SearchAbstractClass
        param_names = []
        param_spaces = []
        for name, hyperparam in param_dict.items():
            if any(
                isinstance(hyperparam, sko_type)
                for sko_type in [Real, Integer, Categorical]
            ):
                param_names.append(name)
                param_spaces.append(hyperparam)
            else:
                raise ValueError(
                    "Unexpected parameter type: {} - {}".format(
                        str(name), str(hyperparam)
                    )
                )

        self.param_names = param_names
        self.param_spaces = param_spaces
        self.param_dict = param_dict

        assert default is dict, f"default param set must be a dictionary"
        for name in default.keys():
            assert name in self.param_names, f"name {name} in default param set not found in param space"
        self.default = default

    def random_samples(self, n, rs, sampler_type="Sobol", sampler_args=None):
        """
        n : (required) number of samples
        rs: (required) numpy random state
        sampler_type: must be key of SAMPLER_DICT
        sampler_args: args passed to skopt sampler

        use sampler functions from skopt.sampler to generate random samples

        """
        if self.param_dict == {}:
            if n != 1:
                raise Exception("no hyperparameters. n must be 1")

            return [{}]

        if sampler_args is None:
            sampler_args = {}

        assert (
            sampler_type in SAMPLER_DICT
        ), f"sampler type {sampler_type} not recognized. sampler_type must be one of {list(SAMPLER_DICT.keys())}."

        sampler = SAMPLER_DICT[sampler_type](**sampler_args)

        # sample hyperparameter values. this returns a list of lists
        samples = sampler.generate(self.param_spaces, n, random_state=rs)

        # put each hyperparameter sample (a list) into its own dict
        return [
            {name: samples[j][i_param] for i_param, name in enumerate(self.param_names)}
            for j in range(n)
        ]
