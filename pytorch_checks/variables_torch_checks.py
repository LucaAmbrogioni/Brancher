import torch.nn as nn
import chainer
from chainer import ChainList

def dummy_fun(input):
    return input

class VarLink(ChainList):

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        links = [link
                 for x in kwargs.values()
                 for link in dummy_fun(x)]
        super().__init__(*links)

    def __call__(self, values):
        return {k: dummy_fun(x) for k, x in self.kwargs.items()}

##
d = {'mu': [chainer.links.Linear(1, 1)], 'sigma': [chainer.links.Linear(1, 1)]}
v = VarLink(**d)
v([1,2])

##
class VarLink2(nn.ModuleList):

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        modules = [module
                 for x in kwargs.values()
                 for module in dummy_fun(x)]
        super().__init__(modules)

    def __call__(self, values):
        return {k: dummy_fun(x) for k, x in self.kwargs.items()}

##
d = {'mu': [nn.Linear(1, 1)], 'sigma': [nn.Linear(1, 1)]}
v = VarLink2(**d)
v([1,2])
