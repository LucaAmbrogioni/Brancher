import numbers

import numpy as np
import torch.nn as nn

import brancher.distributions as distributions
import brancher.functions as BF
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
        self.partial_links = {name: var2link(link) for name, link in kwargs.items()}

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
    def __init__(self, dataset, name, learnable=False, is_observed=False, batch_size=None, indices=None, weights=None): #TODO: Ugly logic
        self._type = "Empirical"
        input_parameters = {"dataset": dataset, "batch_size": batch_size, "indices": indices, "weights": weights}
        ranges = {par_name: geometric_ranges.UnboundedRange()
                  for par_name, par_value in input_parameters.items()
                  if par_value is not None}
        kwargs = {par_name: par_value
                  for par_name, par_value in input_parameters.items()
                  if par_value is not None}
        super().__init__(name, **kwargs, learnable=learnable, ranges=ranges, is_observed=is_observed)

        if not batch_size:
            if indices:
                batch_size = len(indices)
            else:
                raise ValueError("Either the indices or the batch size has to be given as input")

        self.batch_size = batch_size
        self.distribution = distributions.EmpiricalDistribution(batch_size=batch_size, is_observed=is_observed)


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

    def __add__(self, other):
        if isinstance(other, NormalVariable):
            return NormalVariable(self.partial_links["loc"] + other.partial_links["loc"],
                                  scale=BF.sqrt(self.partial_links["scale"]**2 + other.partial_links["scale"]**2),
                                  name=self.name + " + " + other.name, learnable=False)
        else:
            return super().__add__(other)


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
            self.distribution = distributions.BinomialDistribution()
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
            ranges = {"softmax_p": geometric_ranges.UnboundedRange()}
            super().__init__(name, softmax_p=softmax_p, learnable=learnable, ranges=ranges)
            self.distribution = distributions.CategoricalDistribution()
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
    def __init__(self, loc, covariance_matrix=None, precision_matrix=None, cholesky_factor=None, name="Multivariate Normal", learnable=False):
        self._type = "Multivariate Normal"
        if cholesky_factor is not None and covariance_matrix is None and precision_matrix is None:
            ranges = {"loc": geometric_ranges.UnboundedRange(),
                      "cholesky_factor": geometric_ranges.UnboundedRange()}
            super().__init__(name, loc=loc, cholesky_factor=cholesky_factor, learnable=learnable, ranges=ranges)
            self.distribution = distributions.MultivariateNormalDistribution()

        elif cholesky_factor is None and covariance_matrix is not None and precision_matrix is None:
            ranges = {"loc": geometric_ranges.UnboundedRange(),
                      "covariance_matrix": geometric_ranges.PositiveDefiniteMatrix()}
            super().__init__(name, loc=loc, covariance_matrix=covariance_matrix, learnable=learnable, ranges=ranges)
            self.distribution = distributions.MultivariateNormalDistribution()

        elif cholesky_factor is None and covariance_matrix is None and precision_matrix is not None:
            ranges = {"loc": geometric_ranges.UnboundedRange(),
                      "precision_matrix": geometric_ranges.UnboundedRange()}
            super().__init__(name, loc=loc, precision_matrix=precision_matrix, learnable=learnable, ranges=ranges)
            self.distribution = distributions.MultivariateNormalDistribution()

        else:
            raise ValueError("Either covariance_matrix or precision_matrix or"+
                             "cholesky_factor needs to be provided as input")