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

from brancher.utilities import broadcast_and_squeeze_mixed
from brancher.utilities import sum_data_dimensions
from brancher.utilities import get_diagonal
from brancher.utilities import broadcast_parent_values
from brancher.utilities import is_discrete, is_tensor

#TODO: We need asserts checking for the right parameters

class Distribution(ABC):
    """
    Summary
    """
    @abstractmethod
    def _calculate_log_probability(self, x, **parameters):
        pass

    @abstractmethod
    def _get_sample(self, **parameters):
        pass

    @abstractmethod
    def _preprocess_parameters_for_log_prob(self, x, **parameters):
        pass

    @abstractmethod
    def _preprocess_parameters_for_sampling(self, **parameters):
        pass

    @abstractmethod
    def _postprocess_sample(self, sample):
        pass

    def calculate_log_probability(self, x, **parameters):
        x, parameters = self._preprocess_parameters_for_log_prob(x, **parameters)
        log_prob = self._calculate_log_probability(x, **parameters)
        return sum_data_dimensions(log_prob)

    def get_sample(self, **parameters):
        parameters = self._preprocess_parameters_for_sampling(**parameters)
        pre_sample = self._get_sample(**parameters)
        sample = self._postprocess_sample(pre_sample)
        return sample


class ContinuousDistribution(Distribution):
    pass


class DiscreteDistribution(Distribution):
    pass


class UnivariateDistribution(Distribution):
    """
    Summary
    """
    def _preprocess_parameters_for_sampling(self, **parameters):
        parameters = broadcast_and_squeeze_mixed((), parameters)
        return parameters

    def _preprocess_parameters_for_log_prob(self, x, **parameters):
        tuple_x, parameters = broadcast_and_squeeze_mixed(tuple([x]), parameters)
        return tuple_x[0], parameters

    def _postprocess_sample(self, sample):
        return sample


class ImplicitDistribution(Distribution):
    """
    Summary
    """
    def _preprocess_parameters_for_sampling(self, **parameters):
        return parameters

    def _preprocess_parameters_for_log_prob(self, x, **parameters):
        return x, parameters

    def _postprocess_sample(self, sample):
        return sample

    def _calculate_log_probability(self, x, **parameters):
        return torch.tensor(np.zeros((1, 1))) #TODO: Implement some checks here


class MultivariateDistribution(Distribution):
    """
    Summary
    """
    pass


class EmpiricalDistribution(ImplicitDistribution): #TODO: The logic of this cluss is very ugly. It needs to be reworked.
    """
    Summary
    """
    def _get_sample(self, **parameters):
        """
        One line description

        Parameters
        ----------
        Returns
        -------
        Without replacement
        """
        dataset = parameters["dataset"]
        if "indices" not in parameters:
            if "weights" in parameters:
                weights = parameters["weights"]
                p = np.array(weights).astype("float64")
                p = p/np.sum(p)
            else:
                p = None
            if is_tensor(dataset):
                if self.is_observed:
                    dataset_size = dataset.shape[1]
                else:
                    dataset_size = dataset.shape[2]
            else:
                dataset_size = len(dataset)
            if dataset_size < self.batch_size:
                raise ValueError("It is impossible to have more samples than the size of the dataset without replacement")
            if is_discrete(dataset): # TODO: This is for allowing discrete data, temporary?
                indices = np.random.choice(range(dataset_size), size=self.batch_size, replace=False, p=p)
            else:
                indices = [np.random.choice(range(dataset_size), size=self.batch_size, replace=False, p=p)
                           for _ in range(number_samples)] #TODO IMPORTANT!: This needs to be fixed
        else:
            indices = parameters["indices"]

        if is_tensor(dataset):
            if isinstance(indices, list) and isinstance(indices[0], np.ndarray):
                if self.is_observed:
                    sample = torch.cat([dataset[n, k, :].unsqueeze(dim=0) for n, k in enumerate(indices)], dim=0)
                else:
                    sample = torch.cat([dataset[n, :, k, :].unsqueeze(dim=0) for n, k in enumerate(indices)], dim=0)

            elif isinstance(indices, list) and isinstance(indices[0], (int, np.int32, np.int64)):
                if self.is_observed:
                    sample = dataset[:, indices, :]
                else:
                    sample = dataset[:, :, indices, :]
            else:
                raise IndexError("The indices of an empirical variable should be either a list of integers or a list of arrays")
        else:
            sample = list(np.array(dataset)[indices]) # TODO: This is for allowing discrete data, temporary? For julia
        return sample


class NormalDistribution(ContinuousDistribution, UnivariateDistribution):
    """
    Summary
    """

    def _calculate_log_probability(self, x, **parameters):
        """
        One line description

        Parameters
        ----------

        Returns
        -------
        """
        log_prob = distributions.normal.Normal(loc=parameters["loc"],
                                               scale=parameters["scale"]).log_prob(x)
        return log_prob

    def _get_sample(self, **parameters):
        """
        One line description

        Parameters
        ----------

        Returns
        -------
        """
        return distributions.normal.Normal(loc=parameters["loc"],
                                           scale=parameters["scale"]).rsample()


class LogNormalDistribution(ContinuousDistribution, UnivariateDistribution):
    """
    Summary
    """

    def _calculate_log_probability(self, x, **parameters):
        """
        One line description

        Parameters
        ----------

        Returns
        -------
        """
        log_prob = distributions.log_normal.LogNormal(loc=parameters["loc"],
                                                      scale=parameters["scale"]).log_prob(x)
        return log_prob

    def _get_sample(self, **parameters):
        """
        One line description

        Parameters
        ----------

        Returns
        -------
        """
        return distributions.log_normal.LogNormal(loc=parameters["loc"],
                                                  scale=parameters["scale"]).rsample()


class CauchyDistribution(ContinuousDistribution, UnivariateDistribution):
    """
    Summary
    """

    def _calculate_log_probability(self, x, **parameters):
        """
        One line description

        Parameters
        ----------

        Returns
        -------
        """
        log_prob = distributions.cauchy.Cauchy(loc=parameters["loc"],
                                               scale=parameters["scale"]).log_prob(x)
        return log_prob

    def _get_sample(self, **parameters):
        """
        One line description

        Parameters
        ----------

        Returns
        -------
        """
        return distributions.cauchy.Cauchy(loc=parameters["loc"],
                                           scale=parameters["scale"]).rsample()


class LaplaceDistribution(ContinuousDistribution, UnivariateDistribution):
    """
    Summary
    """

    def _calculate_log_probability(self, x, **parameters):
        """
        One line description

        Parameters
        ----------

        Returns
        -------
        """
        log_prob = distributions.laplace.Laplace(loc=parameters["loc"],
                                                 scale=parameters["scale"]).log_prob(x)
        return log_prob

    def _get_sample(self, **parameters):
        """
        One line description

        Parameters
        ----------

        Returns
        -------
        """
        return distributions.laplace.Laplace(loc=parameters["loc"],
                                             scale=parameters["scale"]).rsample()


class BetaDistribution(ContinuousDistribution, UnivariateDistribution):
    """
    Summary
    """

    def _calculate_log_probability(self, x, **parameters):
        """
        One line description

        Parameters
        ----------

        Returns
        -------
        """
        log_prob = distributions.beta.Beta(concentration0=parameters["alpha"],
                                           concentration1=parameters["beta"]).log_prob(x)
        return log_prob

    def _get_sample(self, **parameters):
        """
        One line description

        Parameters
        ----------

        Returns
        -------
        """
        return distributions.beta.Beta(concentration0=parameters["alpha"],
                                       concentration1=parameters["beta"]).rsample()


class BinomialDistribution(UnivariateDistribution, DiscreteDistribution):
    """
    Summary
    """
    def _calculate_log_probability(self, x, **parameters):
        """
        One line description

        Parameters
        ----------

        Returns
        -------
        """
        log_prob = distributions.binomial.Binomial(total_count=parameters["n"],
                                                   probs=parameters["p"]).log_prob(x)
        return log_prob

    def _get_sample(self, **parameters):
        """
        One line description

        Parameters
        ----------

        Returns
        -------
        """
        return distributions.binomial.Binomial(total_count=parameters["n"],
                                               probs=parameters["p"]).sample()


class LogitBinomialDistribution(UnivariateDistribution, DiscreteDistribution):
    """
    Summary
    """
    def _calculate_log_probability(self, x, **parameters):
        """
        One line description

        Parameters
        ----------

        Returns
        -------
        """
        log_prob = distributions.binomial.Binomial(total_count=parameters["n"],
                                                   logits=parameters["logit_p"]).log_prob(x)
        return log_prob

    def _get_sample(self, **parameters):
        """
        One line description

        Parameters
        ----------

        Returns
        -------
        """
        return distributions.binomial.Binomial(total_count=parameters["n"],
                                               logits=parameters["logit_p"]).sample()
