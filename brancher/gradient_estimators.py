"""
Inference
---------
Module description
"""
from abc import ABC, abstractmethod

import numpy as np

import torch

from brancher.config import device
from brancher.utilities import map_iterable


class GradientEstimator(ABC):

    def __init__(self, function, sampler, empirical_samples={}):
        self.function = function
        self.sampler = sampler
        self.empirical_samples = empirical_samples

    @abstractmethod
    def __call__(self, samples):
        pass


class BlackBoxEstimator(GradientEstimator):

    def __call__(self, n_samples):
        samples = self.sampler._get_sample(n_samples, differentiable=False)
        samples.update(self.empirical_samples)
        variational_loss = self.sampler.calculate_log_probability(samples)*(self.function(samples).detach())
        model_loss = self.function(samples)
        return (variational_loss + model_loss).mean()


class PathwiseDerivativeEstimator(GradientEstimator):

    def __call__(self, n_samples):
        samples = self.sampler._get_sample(n_samples, differentiable=True)
        samples.update(self.empirical_samples)
        return self.function(samples).mean()


class Taylor1Estimator(GradientEstimator):

    def __call__(self, n_samples):
        observable_samples = self.sampler._get_sample(1)
        observable_samples.update(self.empirical_samples)
        means = {var: var._get_mean(input_values=observable_samples) for var in self.sampler.variables if not var.is_observed}
        means.update(observable_samples)
        return self.function(means).mean()
