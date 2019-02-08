"""
Variables
---------
Module description
"""

from abc import ABC, abstractmethod
import operator

import numpy as np

from brancher.standard_variables import MultivariateNormalVariable
from brancher.standard_variables import var2link
import brancher.functions as BF


class StochasticProcess(ABC):

    @abstractmethod
    def __call__(self, query_points):
        pass


class GaussianProcess(StochasticProcess):

    def __init__(self, mean_function, covariance_function, name):
        self.mean_function = mean_function
        self.covariance_function = covariance_function
        self.name = name

    def __call__(self, query_points):
        x = var2link(query_points)
        return MultivariateNormalVariable(loc=self.mean_function(x),
                                          covariance_matrix=self.covariance_function(x),
                                          name=self.name + str(x))


class CovarianceFunction(ABC):

    def __init__(self, covariance):
        self.covariance = covariance

    def __call__(self, query_points1, query_points2=None): #TODO: Possible input: arrays, deterministic variables (list?, numeric?)
        if not query_points2:
            query_points2 = query_points1
        query_grid = BF.meshgrid([query_points1, query_points2]) #TODO: This needs to be fixed to include the batch dimension
        return self.covariance(query_grid[0], query_grid[1])

    def __add__(self, other):
        if isinstance(other, CovarianceFunction):
            return CovarianceFunction(covariance=lambda x, y: self.covariance(x, y) + other.covariance(x, y))
        elif isinstance(other, (int, float)):
            return CovarianceFunction(covariance=lambda x, y: self.covariance(x, y) + other)
        else:
            raise ValueError("Only covarianceFunctions and numbers can be summed to CovarianceFunctions")

    def __mul__(self, other):
        if isinstance(other, CovarianceFunction):
            return CovarianceFunction(covariance=lambda x, y: self.covariance(x, y)*other.covariance(x, y))
        elif isinstance(other, (int, float)):
            return CovarianceFunction(covariance=lambda x, y: self.covariance(x, y) * other)
        else:
            raise ValueError("Only covarianceFunctions and numbers can be multiplied with CovarianceFunctions")

    def __rmul__(self, other):
        return self.__mul__(other)


class SquaredExponentialCovariance(CovarianceFunction):

    def __init__(self, scale):
        self.scale = var2link(scale)
        covariance = lambda x, y: BF.exp(-(x-y)**2/(2*scale))
        super().__init__(covariance=covariance)


class MeanFunction(ABC):

    def __init__(self, mean):
        self.mean = mean

    def __call__(self, query_points):
        return self.mean(query_points)

    def _apply_operator(self, other, op):
        """
        Args:

        Returns:
        """
        if isinstance(other, MeanFunction):
            return lambda x: op(self.mean(x), other.mean(x))
        elif isinstance(other, (int, float)):
            return lambda x: op(self.mean(x), other)

    def __add__(self, other):
        return self._apply_operator(other, operator.add)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self._apply_operator(other, operator.sub)

    def __rsub__(self, other):
        return -1*self.__sub__(other)

    def __mul__(self, other):
        return self._apply_operator(other, operator.mul)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return self._apply_operator(other, operator.truediv)

    def __rtruediv__(self, other):
        raise NotImplementedError

    def __pow__(self, other):
        return self._apply_operator(other, operator.pow)

    def __rpow__(self, other):
        raise NotImplementedError

class ConstantMean(MeanFunction):

    def __init__(self, value):
        value = var2link(value)
        mean = lambda x: value
        super().__init__(mean=mean)




