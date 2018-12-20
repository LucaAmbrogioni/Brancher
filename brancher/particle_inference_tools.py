import numpy as np

import chainer
import chainer.functions as F

from brancher.variables import DeterministicVariable


class VoronoiSet(object):

    def __init__(self, particles, cost=lambda x, y: np.sum((x - y)**2)):
        self.cost = cost
        if isinstance(particles, list):
            if isinstance(particles[0], chainer.Variable):
                self.particles = [part.data for part in particles]
            elif isinstance(particles[0], DeterministicVariable):
                self.particles = [part.value[0, 0, :].data for part in particles]
            else:
                raise ValueError("The location of the particles should be either deterministic brancher variables, chainer variables or np.array")
        else:
            raise ValueError("The location of the particles should be inserted as a list of locations")

    def __call__(self, index):
        def truncation_rule(x):
            distances = [self.cost(x, y) for y in self.particles]
            if np.argmin(distances) == index:
                return True
            else:
                return False
        return truncation_rule

