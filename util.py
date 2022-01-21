import numpy as np
from typing import Union
import random


# hmmm...
def Maybe(a_type): return Union[type(None),a_type]


vec3d = np.array


def make_vec(*args): return np.asarray(args)


norm = np.linalg.norm


def normalize(v): return v/norm(v)


def complete_base(base):
    # succesively generate vectors independent from the current set
    # and apply gram-schmidt orthogonalization

    # elements in base will be adjusted to be of the same module as vector

    if type(base[0]) not in (list, tuple, np.ndarray):
        base = [base]

    dim = len(base[0])

    desired_norm = np.linalg.norm(base[0])

    base = [np.array(vector, dtype=float) for vector in base]
    base = list(map(normalize, base))

    while len(base) < dim:
        new_vector = np.array([random.uniform(-1, 1) for _ in range(dim)])
        for vector in base:
            projection_length = np.dot(new_vector, vector)
            new_vector = np.subtract(new_vector, projection_length * vector)
            if np.linalg.norm(new_vector) < 0.1:
                break
        else:
            base.append(normalize(new_vector))
    for vector in base:
        vector *= desired_norm
    return base
