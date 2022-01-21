from functools import cached_property

import opensimplex
import numpy as np


DEFAULT_SEED = 415


class Simplex:
    def __init__(self, frequency, seed=DEFAULT_SEED):
        os_instance = opensimplex.OpenSimplex(seed=seed)
        self.noise = os_instance.noise3d
        self.frequency = frequency

    def at(self, x, y, z):
        return self.noise(self.frequency*x, self.frequency*y, self.frequency*z)


class StackedSimplices:
    def __init__(self, qt_layers, frequencies, amplitudes, seeds=None):
        if seeds is None:
            seeds = [DEFAULT_SEED+i for i in range(qt_layers)]

        self.noises = [opensimplex.OpenSimplex(seed=seed).noise3d for seed in seeds]
        self.frequencies = np.asarray(frequencies)
        self.amplitudes = np.asarray(amplitudes)

    def at(self, x, y, z):
        xs, ys, zs = self.frequencies*float(x), self.frequencies*float(y), self.frequencies*float(z)
        hs = [noise(*x) for noise, *x in zip(self.noises, xs, ys, zs)]
        return np.sum(np.asarray(hs)*self.amplitudes)

    @classmethod
    def functional_amplitude(cls, qt_layers, frequencies, fun, seeds=None):
        amplitudes = [fun(frequency) for frequency in frequencies]
        return cls(qt_layers, frequencies, amplitudes, seeds=seeds)

    @cached_property
    def amplitude(self):
        return np.sum(self.amplitudes)

    @cached_property
    def lipschitz_bound(self):
        return np.sum(self.amplitudes*self.frequencies)

    @cached_property
    def highest_frequency(self):
        return np.max(self.frequencies)
