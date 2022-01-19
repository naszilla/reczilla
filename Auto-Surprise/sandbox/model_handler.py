from auto_surprise.algorithms.spaces import HPO_SPACE_MAP
from auto_surprise.constants import SURPRISE_ALGORITHM_MAP

from custom_algorithms import CONSTANT_ALG_SPACE, ConstantAlg

SURPRISE_ALGORITHMS = list(SURPRISE_ALGORITHM_MAP.keys())

# define custom algorithms and parameter spaces
CUSTOM_ALGORITHM_MAP = {"constant": ConstantAlg}
CUSTOM_HPO_SPACES = {"constant": CONSTANT_ALG_SPACE}

ALL_ALGORITHMS = SURPRISE_ALGORITHM_MAP
for name, alg in CUSTOM_ALGORITHM_MAP.items():
    if name in ALL_ALGORITHMS:
        raise Exception(f"algorithm name {name} is already a surprise algorithm")
    ALL_ALGORITHMS[name] = alg

ALL_SPACES = HPO_SPACE_MAP
for name, space in CUSTOM_HPO_SPACES.items():
    if name in ALL_SPACES:
        raise Exception(f"algorithm name {name} is already a surprise algorithm")
    ALL_SPACES[name] = space
