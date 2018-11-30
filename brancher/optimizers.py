"""
Optimizers
---------
Module description
"""
from abc import ABC, abstractmethod
import copy

from chainer import optimizers, Link, Chain, ChainList

from brancher.chains import EmptyChain
from brancher.variables import BrancherClass, Variable, ProbabilisticModel


PO_DEFAULT_APLHA = 0.001
PO_DEFAULT_BETA1 = 0.9
PO_DEFAULT_BETA2 = 0.999
PO_DEFAULT_EPS = 1e-08


class ProbabilisticOptimizer(ABC):
    """
    Summary

    Parameters
    ----------
    optimizer : chainer optimizer
        Summary
    """
    def __init__(self, model, optimizer=None):
        if optimizer is None:
            optimizer = self._get_default_optimizer()
        else:
            self.optimizer = copy.deepcopy(optimizer)
        self.link_set = set()
        self.chain = None
        self.setup(model)
        pass

    @staticmethod
    def _get_default_optimizer(self, **kwargs):
        optimizer = optimizers.Adam(alpha=PO_DEFAULT_APLHA, beta1=PO_DEFAULT_BETA1,
                                    beta2=PO_DEFAULT_BETA2, eps=PO_DEFAULT_EPS)
        return optimizer

    def _update_link_set(self, random_variable):
        assert isinstance(random_variable, BrancherClass)
        link = random_variable.link if hasattr(random_variable, 'link') else None
        if isinstance(link, Link) or isinstance(link, Chain) or isinstance(link, ChainList):
            self.link_set.add(link)

        vars_attr = 'variables' if isinstance(random_variable, ProbabilisticModel) else 'parents'
        for var in getattr(random_variable, vars_attr):
            self._update_link_set(var)

    def setup(self, random_variable): # TODO: Work in progress
        """
        Summary
        """
        self.chain = EmptyChain()
        self._update_link_set(random_variable)
        for link in self.link_set:
            if isinstance(link, Link):
                self.chain.add_link(link)
            elif isinstance(link, ChainList):
                [self.chain.add_link(l) for l in link]

        self.optimizer.setup(self.chain)

    def update(self):
        self.optimizer.update()
