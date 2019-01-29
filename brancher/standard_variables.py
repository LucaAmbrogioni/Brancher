import numbers

import numpy as np
import torch.nn as nn

import brancher.distributions as distributions
import brancher.geometric_ranges as geometric_ranges
from brancher.variables import var2link, Variable, DeterministicVariable, RandomVariable, PartialLink
from brancher.utilities import join_sets_list


class LinkConstructor(nn.ModuleList):
    """
    Summary

    Parameters
    ----------
    """
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        modules = [link
                 for partial_link in kwargs.values()
                 for link in var2link(partial_link).links]
        super().__init__(modules) #TODO: asserts that specified links are valid pytorch modules

    def __call__(self, values):
        return {k: var2link(x).fn(values) for k, x in self.kwargs.items()}


class VariableConstructor(RandomVariable):
    """
    Summary

    Parameters
    ----------
    """
    def __init__(self, name, learnable, ranges, is_observed=False, **kwargs):
        self.name = name
        self._evaluated = False
        self._observed = is_observed
        self._observed_value = None
        self._current_value = None
        self.construct_deterministic_parents(learnable, ranges, kwargs)
        self.parents = join_sets_list([var2link(x).vars for x in kwargs.values()])
        self.link = LinkConstructor(**kwargs)
        self.samples = []
        self.ranges = {}
        self.dataset = None
        self.has_random_dataset = False
        self.has_observed_value = False
        self.is_normalized = True


    def construct_deterministic_parents(self, learnable, ranges, kwargs):
        for parameter_name, value in kwargs.items():
            if not isinstance(value, (Variable, PartialLink)):
                if isinstance(value, np.ndarray):
                    dim = value.shape[0] #TODO: This is probably not general enough
                elif isinstance(value, numbers.Number):
                    dim = 1
                else:
                    dim = [] #TODO: You should consider the other possible cases individually
                deterministic_parent = DeterministicVariable(ranges[parameter_name].inverse_transform(value, dim),
                                                             self.name + "_" + parameter_name, learnable, is_observed=self._observed)
                kwargs.update({parameter_name: ranges[parameter_name].forward_transform(deterministic_parent, dim)})


class EmpiricalVariable(VariableConstructor):
    """
    Summary

    Parameters
    ----------
    """
    def __init__(self, dataset, name, learnable=False, is_observed=False, batch_size=(), indices=(), weights=()):
        self._type = "Empirical"
        ranges = {"dataset": geometric_ranges.UnboundedRange(),
                  "batch_size": geometric_ranges.UnboundedRange(),
                  "indices": geometric_ranges.UnboundedRange(),
                  "weights": geometric_ranges.UnboundedRange()}
        super().__init__(name, dataset=dataset, indices=indices, weights=weights,
                         learnable=learnable, ranges=ranges, is_observed=is_observed)
        self.distribution = distributions.EmpiricalDistribution()
        self.distribution.is_observed = is_observed #TODO: Clean up here?
        if batch_size:
            self.distribution.batch_size = batch_size
            self.batch_size = batch_size
        elif indices:
            self.distribution.batch_size = len(indices)
            self.batch_size = batch_size #TODO: Clean up here?
        else:
            raise ValueError("Either the indices or the batch size has to be given as input")


class RandomIndices(EmpiricalVariable):
    """
    Summary

    Parameters
    ----------
    """
    def __init__(self, dataset_size, batch_size, name, is_observed=False):
        self._type = "Random Index"
        super().__init__(dataset=list(range(dataset_size)),
                         batch_size=batch_size, is_observed=is_observed, name=name)

    def __len__(self):
        return self.batch_size


class NormalVariable(VariableConstructor):
    """
    Summary

    Parameters
    ----------
    """
    def __init__(self, loc, scale, name, learnable=False):
        self._type = "Normal"
        ranges = {"loc": geometric_ranges.UnboundedRange(),
                  "scale": geometric_ranges.RightHalfLine(0.)}
        super().__init__(name, loc=loc, scale=scale, learnable=learnable, ranges=ranges)
        self.distribution = distributions.NormalDistribution()


class CauchyVariable(VariableConstructor):
    """
    Summary

    Parameters
    ----------
    """
    def __init__(self, loc, scale, name, learnable=False):
        self._type = "Cauchy"
        ranges = {"loc": geometric_ranges.UnboundedRange(),
                  "scale": geometric_ranges.RightHalfLine(0.)}
        super().__init__(name, loc=loc, scale=scale, learnable=learnable, ranges=ranges)
        self.distribution = distributions.CauchyDistribution()


class LaplaceVariable(VariableConstructor):
    """
    Summary

    Parameters
    ----------
    """
    def __init__(self, loc, scale, name, learnable=False):
        self._type = "Laplace"
        ranges = {"loc": geometric_ranges.UnboundedRange(),
                  "scale": geometric_ranges.RightHalfLine(0.)}
        super().__init__(name, loc=loc, scale=scale, learnable=learnable, ranges=ranges)
        self.distribution = distributions.LaplaceDistribution()


class LogNormalVariable(VariableConstructor):
    """
    Summary

    Parameters
    ----------
    """
    def __init__(self, loc, scale, name, learnable=False):
        self._type = "Log Normal"
        ranges = {"loc": geometric_ranges.UnboundedRange(),
                  "scale": geometric_ranges.RightHalfLine(0.)}
        super().__init__(name, loc=loc, scale=scale, learnable=learnable, ranges=ranges)
        self.distribution = distributions.LogNormalDistribution()


class LogitNormalVariable(VariableConstructor):
    """
    Summary

    Parameters
    ----------
    """
    def __init__(self, loc, scale, name, learnable=False):
        self._type = "Logit Normal"
        ranges = {"loc": geometric_ranges.UnboundedRange(),
                  "scale": geometric_ranges.RightHalfLine(0.)}
        super().__init__(name, loc=loc, scale=scale, learnable=learnable, ranges=ranges)
        self.distribution = distributions.LogitNormalDistribution()


class BetaVariable(VariableConstructor):
    """
    Summary

    Parameters
    ----------
    """
    def __init__(self, alpha, beta, name, learnable=False):
        self._type = "Logit Normal"
        ranges = {"alpha": geometric_ranges.RightHalfLine(0.),
                  "beta": geometric_ranges.RightHalfLine(0.)}
        super().__init__(name, alpha=alpha, beta=beta, learnable=learnable, ranges=ranges)
        self.distribution = distributions.BetaDistribution()


class BinomialVariable(VariableConstructor):
    """
    Summary

    Parameters
    ----------
    """
    def __init__(self, n, p=None, logit_p=None, name="Binomial", learnable=False):
        self._type = "Binomial"
        if p is not None and logit_p is None:
            ranges = {"n": geometric_ranges.UnboundedRange(),
                      "p": geometric_ranges.Interval(0., 1.)}
            super().__init__(name, n=n, p=p, learnable=learnable, ranges=ranges)
            self.distribution = distributions.BinomialDistribution()
        elif logit_p is not None and p is None:
            ranges = {"n": geometric_ranges.UnboundedRange(),
                      "logit_p": geometric_ranges.UnboundedRange()}
            super().__init__(name, n=n, logit_p=logit_p, learnable=learnable, ranges=ranges)
            self.distribution = distributions.LogitBinomialDistribution()
        else:
            raise ValueError("Either p or " +
                             "logit_p needs to be provided as input")


class CategoricalVariable(VariableConstructor): #TODO: Work in progress
    """
    Summary

    Parameters
    ----------
    """
    def __init__(self, p=None, softmax_p=None, name="Categorical", learnable=False):
        self._type = "Categorical"
        if p is not None and softmax_p is None:
            ranges = {"p": geometric_ranges.Simplex()}
            super().__init__(name, p=p, learnable=learnable, ranges=ranges)
            self.distribution = distributions.CategoricalDistribution()
        elif softmax_p is not None and p is None:
            ranges = {"z": geometric_ranges.UnboundedRange()}
            super().__init__(name, z=softmax_p, learnable=learnable, ranges=ranges)
            self.distribution = distributions.SoftmaxCategoricalDistribution()
        else:
            raise ValueError("Either p or " +
                             "softmax_p needs to be provided as input")


class ConcreteVariable(VariableConstructor):
    """
    Summary

    Parameters
    ----------
    """
    def __init__(self, tau, p, name, learnable=False):
        self._type = "Concrete"
        ranges = {"tau": geometric_ranges.RightHalfLine(0.),
                  "p": geometric_ranges.Simplex()}
        super().__init__(name, tau=tau, p=p, learnable=learnable, ranges=ranges)
        self.distribution = distributions.ConcreteDistribution()


class MultivariateNormalVariable(VariableConstructor):
    """
    Summary

    Parameters
    ----------
    """
    def __init__(self, mu, cov=None, chol_cov=None, diag_cov=None, name="Multivariate Normal", learnable=False):
        self._type = "Multivariate Normal"
        if chol_cov is not None and diag_cov is None:
            ranges = {"mu": geometric_ranges.UnboundedRange(),
                      "chol_cov": geometric_ranges.UnboundedRange()}
            super().__init__(name, mu=mu, chol_cov=chol_cov, learnable=learnable, ranges=ranges)
            self.distribution = distributions.CholeskyMultivariateNormal()
        elif diag_cov is not None and chol_cov is None:
            ranges = {"mean": geometric_ranges.UnboundedRange(),
                      "var": geometric_ranges.RightHalfLine(0.)}
            super().__init__(name, mean=mu, var=diag_cov, learnable=learnable, ranges=ranges)
            self.distribution = distributions.NormalDistribution()
        else:
            raise ValueError("Either chol_cov (cholesky factor of the covariance matrix) or "+
                             "diag_cov (diagonal of the covariance matrix) need to be provided as input")