"""
Distributions
---------
Module description
"""
from abc import ABC, abstractmethod

import chainer
import chainer.functions as F
import numpy as np
from scipy.special import binom

from brancher.utilities import broadcast_and_squeeze
from brancher.utilities import sum_data_dimensions
from brancher.utilities import get_diagonal


class Distribution(ABC):
    """
    Summary
    """
    @abstractmethod
    def calculate_log_probability(self, *parameters):
        pass

    @abstractmethod
    def get_sample(self, *parameters):
        pass


## Implicit distributions ##
class ImplicitDistribution(Distribution):

    def calculate_log_probability(self, x, **parameters):
        return np.zeros((1, 1)).astype("float32") #TODO: Implement some checks here


class EmpiricalDistribution(ImplicitDistribution):
    """
    Summary
    """
    def get_sample(self, dataset, indices, number_samples):
        """
        One line description

        Parameters
        ----------
        Returns
        -------
        Without replacement
        """
        if not indices:
            if isinstance(dataset, chainer.Variable):
                dataset_size = dataset.shape[1]
            else:
                dataset_size = len(dataset)
            if dataset_size < number_samples:
                raise ValueError("It is impossible to have more samples than the size of the dataset without replacement")
            indices = np.random.choice(range(dataset_size), size=self.batch_size, replace=False)
        if isinstance(dataset, chainer.Variable):
            sample = dataset[:, indices, :]
        else:
            sample = list(np.array(dataset)[indices]) #TODO: clean up
        return sample


## Univariate distributions ##
class UnivariateDistribution(Distribution):
    pass


class NormalDistribution(UnivariateDistribution):
    """
    Summary
    """
    def calculate_log_probability(self, x, mean, var):
        """
        One line description

        Parameters
        ----------

        Returns
        -------
        """
        x, mean, var = broadcast_and_squeeze(x, mean, var)
        log_probability = -0.5*F.log(2*np.pi*var) - 0.5*(x-mean)**2/var
        return sum_data_dimensions(log_probability)

    def get_sample(self, mean, var, number_samples):
        """
        One line description

        Parameters
        ----------

        Returns
        -------
        """
        mean, var = broadcast_and_squeeze(mean, var)
        sample = mean + F.sqrt(var)*np.random.normal(0, 1, size=mean.shape)
        return sample


class CauchyDistribution(UnivariateDistribution):
    """
    Summary
    """
    def calculate_log_probability(self, x, mu, sigma):
        """
        One line description

        Parameters
        ----------

        Returns
        -------
        """
        x, mu, sigma = broadcast_and_squeeze(x, mu, sigma)
        log_probability = -F.log(1 + (x-mu)**2/sigma**2)
        return sum_data_dimensions(log_probability)

    def get_sample(self, mu, sigma, number_samples):
        """
        One line description

        Parameters
        ----------

        Returns
        -------
        """
        mu, sigma = broadcast_and_squeeze(mu, sigma)
        sample = mu + sigma*F.tan(np.pi*np.random.uniform(0,1,size=mu.shape).astype(np.float32))
        return sample


class LogNormalDistribution(UnivariateDistribution):
    """
    Summary
    """
    def calculate_log_probability(self, x, mu, sigma):
        """
        One line description

        Parameters
        ----------

        Returns
        -------
        """
        x, mu, sigma = broadcast_and_squeeze(x, mu, sigma)
        log_probability = -0.5*np.log(2*np.pi) - F.log(x) - F.log(sigma) - 0.5*(F.log(x)-mu)**2/(sigma**2)
        return sum_data_dimensions(log_probability)

    def get_sample(self, mu, sigma, number_samples):
        """
        One line description

        Parameters
        ----------

        Returns
        -------
        """
        mu, sigma = broadcast_and_squeeze(mu, sigma)
        log_sample = mu + sigma*np.random.normal(0,1,size=mu.shape)
        return F.exp(log_sample)


class LogitNormalDistribution(UnivariateDistribution):
    """
    Summary
    """
    def calculate_log_probability(self, x, mu, sigma):
        """
        One line description

        Parameters
        ----------

        Returns
        -------
        """
        def logit(p):
            return F.log(p) - F.log(1-p)
        x, mu, sigma = broadcast_and_squeeze(x, mu, sigma)
        log_probability = -0.5*np.log(2*np.pi) - F.log(x) - F.log(1-x) - F.log(sigma) - 0.5*(logit(x)-mu)**2/(sigma**2)
        return sum_data_dimensions(log_probability)

    def get_sample(self, mu, sigma, number_samples):
        """
        One line description

        Parameters
        ----------

        Returns
        -------
        """
        mu, sigma = broadcast_and_squeeze(mu, sigma)
        logit_sample = mu + sigma*np.random.normal(0,1,size=mu.shape)
        return F.sigmoid(logit_sample)


class BinomialDistribution(UnivariateDistribution):
    """
    Summary
    """
    def calculate_log_probability(self, x, n, p):
        """
        One line description

        Parameters
        ----------

        Returns
        -------
        """
        x, n, p = broadcast_and_squeeze(x, n, p)
        x, n = x.data, n.data
        log_probability = np.log(binom(n, x)) + x*F.log(p) + (n-x)*F.log(1-p)
        return sum_data_dimensions(log_probability)

    def get_sample(self, n, p, number_samples):
        """
        One line description

        Parameters
        ----------

        Returns
        -------
        """
        n, p = broadcast_and_squeeze(n, p)
        binomial_sample = np.random.binomial(n.data, p.data) #TODO: Not reparametrizable (Gumbel?)
        return chainer.Variable(binomial_sample.astype("int32"))


class LogitBinomialDistribution(UnivariateDistribution):
    """
    Summary
    """
    def calculate_log_probability(self, x, n, z):
        """
        One line description

        Parameters
        ----------

        Returns
        -------
        """
        x, n, z = broadcast_and_squeeze(x, n, z)
        x, n = x.data, n.data
        alpha = F.relu(-z).data
        beta = F.relu(z).data
        success_term = x*alpha - x*F.log(np.exp(alpha) + F.exp(alpha-z))
        failure_term = (n-x)*beta - (n-x)*F.log(np.exp(beta) + F.exp(beta+z))
        log_probability = np.log(binom(n, x)) + success_term + failure_term
        return sum_data_dimensions(log_probability)

    def get_sample(self, n, z, number_samples):
        """
        One line description

        Parameters
        ----------

        Returns
        -------
        """
        n, z = broadcast_and_squeeze(n, z)
        binomial_sample = np.random.binomial(n.data, F.sigmoid(z).data) #TODO: Not reparametrizable (Gumbel?)
        return chainer.Variable(binomial_sample.astype("int32"))


## Multivariate distributions ##
class MultivariateDistribution(Distribution):
    pass


class CholeskyMultivariateNormal(MultivariateDistribution): #TODO: This needs to be finished
    """
    Summary
    """

    def calculate_log_probability(self, x, mu, chol_cov):
        """
        One line description

        Parameters
        ----------

        Returns
        -------
        """
        log_det = 2*F.sum(F.log(get_diagonal(chol_cov)), axis=2)
        whitened_input = F.matmul(F.transpose(chol_cov, axes=(1, 2, 4, 3)), x)
        exponent = F.sum(whitened_input**2, axis=2)
        log_probability = -0.5*np.log(2*np.pi) -0.5*log_det -0.5*exponent
        return sum_data_dimensions(log_probability)

    def get_sample(self, mu, chol_cov, number_samples):
            """
            One line description

            Parameters
            ----------

            Returns
            -------
            """
            random_vector = np.random.normal(0,1,size=mu.shape).astype("float32")
            return mu + F.matmul(chol_cov, random_vector)


class ConcreteDistribution(MultivariateDistribution):
    """
    Summary
    """
    def calculate_log_probability(self, x, p, tau):
        """
        One line description

        Parameters
        ----------

        Returns
        -------
        """
        if p.shape > 2:
            raise ValueError("The parameter p of the concrete distribution should be a 2D array")
        dim = p.shape[1]
        p, tau = F.broadcast(p, tau)
        normalization = F.log(F.sum(p*x**(-tau-1), axis=1))
        log_probability = (np.log(dim + 1) + (dim - 1)*F.log(tau) + F.sum(F.log(p) + (-tau + 1)*F.log(x), axis=1)
                           - dim*normalization)
        return log_probability

    def get_sample(self, p, tau, number_samples):
        """
        One line description

        Parameters
        ----------

        Returns
        -------
        """
        p, tau = F.broadcast(p, tau)
        gumbel_sample = np.random.gumbel(0, 1, size=p.shape)
        return F.softmax((F.log(p) + gumbel_sample)/tau, axis=2)


# StochasticProcesses #
class StochasticProcesses(MultivariateDistribution):
    pass


class DiffusionProcess(StochasticProcesses): #TODO: This needs to be finished
    """
    Summary
    """
    def __init__(self, drift_function, diffusion_function, time_spacing):
        self.drift = drift_function
        self.diffusion = diffusion_function
        self.time_spacing = time_spacing

    def calculate_log_probability(self, time_series):
        """
        One line description

        Parameters
        ----------

        Returns
        -------
        """
        pass

    def get_sample(self, p, tau, number_samples):
        """
        One line description

        Parameters
        ----------
        mu : numeric
        sigma : numeric
        number_samples : int

        Returns
        -------
        sample : type
        """
        pass
