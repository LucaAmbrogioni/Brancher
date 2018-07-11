import numbers
import numpy as np
import chainer

import brancher.distributions as distributions
import brancher.geometric_ranges as geometric_ranges
from brancher.variables import var2link, Variable, DeterministicVariable, RandomVariable, PartialLink
from brancher.utilities import join_sets_list
import brancher.functions as BF


class VariableConstructor(RandomVariable):
    """
    Summary

    Parameters
    ----------
    """
    def __init__(self, name, learnable, ranges, **kwargs):

        class VarLink(chainer.ChainList):

            def __init__(self):
                self.kwargs = kwargs
                links = [link
                         for partial_link in kwargs.values()
                         for link in var2link(partial_link).links]
                super().__init__(*links)

            def __call__(self, values):
                return {k: var2link(x).fn(values) for k, x in self.kwargs.items()}

        self.name = name
        self.construct_deterministic_parents(learnable, ranges, kwargs)
        self.parents = join_sets_list([var2link(x).vars for x in kwargs.values()])
        self.link = VarLink()
        self.samples = []
        self.ranges = {}

        self._evaluated = False
        self._observed = False
        self._observed_value = None
        self._current_value = None

    def construct_deterministic_parents(self, learnable, ranges, kwargs):
        for parameter_name, value in kwargs.items():
            if not isinstance(value, (Variable, PartialLink)):
                if isinstance(value, np.ndarray):
                    dim = value.shape[1]
                elif isinstance(value, numbers.Number):
                    dim = 1
                else:
                    raise TypeError("The input value should be either a number or a np.ndarray")
                deterministic_parent = DeterministicVariable(ranges[parameter_name].inverse_transform(value, dim),
                                                             self.name + "_" + parameter_name, learnable)
                kwargs.update({parameter_name: ranges[parameter_name].forward_transform(deterministic_parent, dim)})


class NormalVariable(VariableConstructor):
    """
    Summary

    Parameters
    ----------
    """
    def __init__(self, mean, var, name, learnable=False):
        ranges = {"mean": geometric_ranges.UnboundedRange(),
                  "var": geometric_ranges.RightHalfLine(0.)}
        super().__init__(name, mean=mean, var=var, learnable=learnable, ranges=ranges)
        self.distribution = distributions.NormalDistribution()


class CauchyVariable(VariableConstructor):
    """
    Summary

    Parameters
    ----------
    """
    def __init__(self, mu, sigma, name, learnable=False):
        ranges = {"mu": geometric_ranges.UnboundedRange(),
                  "sigma": geometric_ranges.RightHalfLine(0.)}
        super().__init__(name, mu=mu, sigma=sigma, learnable=learnable, ranges=ranges)
        self.distribution = distributions.CauchyDistribution()


class LogNormalVariable(VariableConstructor):
    """
    Summary

    Parameters
    ----------
    """
    def __init__(self, mu, sigma, name, learnable=False):
        ranges = {"mu": geometric_ranges.UnboundedRange(),
                  "sigma": geometric_ranges.RightHalfLine(0.)}
        super().__init__(name, mu=mu, sigma=sigma, learnable=learnable, ranges=ranges)
        self.distribution = distributions.LogNormalDistribution()


class LogitNormalVariable(VariableConstructor):
    """
    Summary

    Parameters
    ----------
    """
    def __init__(self, mu, sigma, name, learnable=False):
        ranges = {"mu": geometric_ranges.UnboundedRange(),
                  "sigma": geometric_ranges.RightHalfLine(0.)}
        super().__init__(name, mu=mu, sigma=sigma, learnable=learnable, ranges=ranges)
        self.distribution = distributions.LogitNormalDistribution()


class BinomialVariable(VariableConstructor):
    """
    Summary

    Parameters
    ----------
    """
    def __init__(self, n, p=None, logit_p=None, name="Binomial", learnable=False):
        if p is not None and logit_p is None:
            ranges = {"n": geometric_ranges.UnboundedRange(),
                      "p": geometric_ranges.Interval(0., 1.)}
            super().__init__(name, n=n, p=p, learnable=learnable, ranges=ranges)
            self.distribution = distributions.BinomialDistribution()
        elif logit_p is not None and p is None:
            ranges = {"n": geometric_ranges.UnboundedRange(),
                      "z": geometric_ranges.UnboundedRange()}
            super().__init__(name, n=n, z=logit_p, learnable=learnable, ranges=ranges)
            self.distribution = distributions.LogitBinomialDistribution()
        else:
            raise ValueError("Either p or " +
                             "logit_p need to be provided as input")


class ConcreteVariable(VariableConstructor):
    """
    Summary

    Parameters
    ----------
    """
    def __init__(self, tau, p, name, learnable=False):
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