"""
Distributions
---------
Module description
"""
from abc import ABC, abstractmethod
from collections.abc import Iterable
import copy

import numpy as np
from scipy.special import binom

import torch
from torch import distributions

from brancher.utilities import broadcast_and_squeeze
from brancher.utilities import sum_data_dimensions
from brancher.utilities import get_diagonal
from brancher.utilities import broadcast_parent_values
from brancher.utilities import is_discrete, is_tensor

# TODO: This module is messy with ad hoc solutions for every distribution. You need to make everything more standardized.

# TODO: You need to implement a consistent broadcasting strategy for univariate, multivariate and tensor distributions.

# TODO: Remove "number samples" input


class Distribution(ABC):
    """
    Summary
    """
    @abstractmethod
    def _calculate_log_probability(self, x, *parameters):
        pass

    @abstractmethod
    def _get_sample(self, *parameters):
        pass

    @abstractmethod
    def _preprocess_parameters_for_log_prob(self, x, *parameters):
        pass

    @abstractmethod
    def _preprocess_parameters_for_sampling(self, *parameters):
        pass

    @abstractmethod
    def _postprocess_sample(self, *parameters):
        pass

    def calculate_log_probability(self, x, *parameters):
        data_and_parameters = self._preprocess_parameters_for_log_prob(x, *parameters)
        log_prob = self._calculate_log_probability(*data_and_parameters[1:])
        return sum_data_dimensions(log_prob)

    def get_sample(self, *parameters):
        parameters = self._preprocess_parameters_for_sampling(*parameters)
        pre_sample = self._get_sample(*parameters)
        sample = self._postprocess_sample(pre_sample)
        return sample


class ContinuousDistribution(Distribution):
    """
    Summary
    """
    def _preprocess_parameters_for_sampling(self, *parameters):
        parameters = broadcast_and_squeeze(*parameters)
        return  parameters

    def _preprocess_parameters_for_log_prob(self, x, *parameters):
        data_and_parameters = broadcast_and_squeeze(x, *parameters)
        return data_and_parameters

class DiscreteDistribution(Distribution):
    """
    Summary
    """
    pass


class UnivariateDistribution(Distribution):
    """
    Summary
    """

    pass


class MultivariateDistribution(Distribution):
    """
    Summary
    """
    pass


class NormalDistribution(ContinuousDistribution, UnivariateDistribution):
    """
    Summary
    """

    def _calculate_log_probability(self, x, mu, sigma):
        """
        One line description

        Parameters
        ----------

        Returns
        -------
        """
        log_prob = distributions.normal.Normal(loc=mu, scale=sigma).log_prob(x)
        return log_prob

    def _get_sample(self, mu, sigma, number_samples):
        """
        One line description

        Parameters
        ----------

        Returns
        -------
        """
        return distributions.normal.Normal(loc=mu, scale=sigma).rsample()


class LogNormalDistribution(ContinuousDistribution, UnivariateDistribution):
    """
    Summary
    """

    def _calculate_log_probability(self, x, mu, sigma):
        """
        One line description

        Parameters
        ----------

        Returns
        -------
        """
        log_prob = distributions..log_normal.LogNormal(loc=mu, scale=sigma).log_prob(x)
        return log_prob

    def _get_sample(self, mu, sigma, number_samples):
        """
        One line description

        Parameters
        ----------

        Returns
        -------
        """
        return distributions..log_normal.LogNormal(loc=mu, scale=sigma).rsample()