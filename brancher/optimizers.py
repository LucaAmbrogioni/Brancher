"""
Optimizers
---------
Module description
"""
from abc import ABC, abstractmethod
from collections.abc import Iterable
import copy

import torch

from brancher.standard_variables import LinkConstructor
from brancher.modules import ParameterModule, EmptyModule
from brancher.variables import BrancherClass, Variable, ProbabilisticModel


class ProbabilisticOptimizer(ABC):
    """
    Summary

    Parameters
    ----------
    optimizer : chainer optimizer
        Summary
    """
    def __init__(self, model, optimizer='Adam', **kwargs):
        assert isinstance(optimizer, str), 'Optimizer should be a name of available pytoch optimizers' #TODO: improve, list optim?
        self.link_set = set()
        self.module = None
        self.setup(model, optimizer, **kwargs) #TODO: add assers for checking the params dictionary

    # @staticmethod
    # def _get_default_optimizer(self, **kwargs):
    #     optimizer = torch.optim.Adam(lr=PO_DEFAULT_APLHA, betas=(PO_DEFAULT_BETA1, PO_DEFAULT_BETA2), eps=PO_DEFAULT_EPS)
    #     return optimizer

    def _update_link_set(self, random_variable): #TODO: rename as just variable, because can be deterministic (Parameter)
        assert isinstance(random_variable, BrancherClass) #TODO: add intuitive error
        link = random_variable.link if hasattr(random_variable, 'link') else None
        #if isinstance(link, Link) or isinstance(link, Chain) or isinstance(link, ChainerList):
        if isinstance(link, (ParameterModule, LinkConstructor)): #TODO: make sure that if user inputs nn.ModuleList, this works
            self.link_set.add(link)

        vars_attr = 'variables' if isinstance(random_variable, ProbabilisticModel) else 'parents'
        for var in getattr(random_variable, vars_attr):
            self._update_link_set(var)

    def add_variable2module(self, random_variable):
        """
        Summary
        """
        self._update_link_set(random_variable)
        for link in self.link_set:
            if isinstance(link, ParameterModule):
                self.module.append(link)
            elif isinstance(link, LinkConstructor):
                [self.module.append(l) for l in link]

    def setup(self, model, optimizer, **kwargs):
        self.module = EmptyModule() #TODO: better name: aggregation of all links in all variables of the model
        optimizer_class = getattr(torch.optim, optimizer)
        if isinstance(model, (Variable, ProbabilisticModel)):
            self.add_variable2module(model)
        elif isinstance(model, Iterable) and all([isinstance(submodel, (Variable, ProbabilisticModel))
                                                  for submodel in model]):
            [self.add_variable2module(submodel) for submodel in model]
        else:
            raise ValueError("Only brancher variables and iterable of variables can be added to a probabilistic optimizer")
        if  list(self.module.parameters()):
            self.optimizer = optimizer_class(self.module.parameters(), **kwargs)
        else:
            self.optimizer = None


    def update(self):
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()