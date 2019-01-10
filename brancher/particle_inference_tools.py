import numpy as np

import chainer
import chainer.functions as F

from brancher.variables import DeterministicVariable


class VoronoiSet(object):

    def __init__(self, particles, cost=lambda x, y: np.sum((x - y)**2)):
        self.cost = cost
        self.particles = particles
        self.locations = None
        self.update_locations()

    def update_locations(self):
        if isinstance(self.particles, list):
            if isinstance(self.particles[0], chainer.Variable):
                self.locations = [part.data for part in self.particles]
            elif isinstance(self.particles[0], DeterministicVariable):
                self.locations = [part.value[0, 0, :].data for part in self.particles]
            else:
                raise ValueError("The location of the particles should be either deterministic brancher variables, chainer variables or np.array")
        else:
            raise ValueError("The location of the particles should be inserted as a list of locations")

    def __call__(self, x, index):
        self.update_locations()
        distances = [self.cost(x, y) for y in self.locations]
        if np.argmin(distances) == index:
            return True
        else:
            return False

