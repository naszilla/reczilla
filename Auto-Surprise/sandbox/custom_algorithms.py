from surprise import AlgoBase
from hyperopt import hp

LB = 0
UB = 10

class ConstantAlg(AlgoBase):
    """
    template for custom surprise algorithms

    see the surprise docs for advice on custom algorithms:
    https://surprise.readthedocs.io/en/stable/building_custom_algo.html
    """

    def __init__(self, constant=0):

        # Always call base method before doing anything.
        AlgoBase.__init__(self)

        self.constant = constant

    def estimate(self, u, i):

        return self.constant

CONSTANT_ALG_SPACE = {
    "constant": hp.uniform("constant", LB, UB)
}