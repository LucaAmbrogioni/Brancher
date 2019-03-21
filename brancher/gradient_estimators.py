"""
Inference
---------
Module description
"""
from abc import ABC, abstractmethod

import numpy as np

import torch

from brancher.config import device


class GradientEstimator(ABC):

    def __init__(self, function, sampler, perturbative_correction=False):
        self.perturbative_correction = perturbative_correction
        self.function = function
        self.sampler = sampler

    @abstractmethod
    def __call__(self, samples):
        pass

class BlackBoxEstimator(GradientEstimator):

    def __cal__(self, samples):
        return self.sampler.calculate_log_probability(input_vales=samples)*self.function(samples.detach()).detach()


class PathwiseDerivativeEstimator(GradientEstimator):

    def __cal__(self, samples):
        return self.function(samples.detach())


class PerturbativeEstimator(GradientEstimator):
    pass


