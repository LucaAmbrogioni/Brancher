"""
NN.modules
---------
Module description
"""
#from chainer import ChainList
from torch import nn

class ParameterModule(nn.Module):
    """
    Summary
    """
    def __init__(self, parameter):
        links = []
        super(ParameterModule, self).__init__()
        self.parameter = parameter

    def __call__(self, *args, **kwargs):
        return self.parameter

class EmptyModule(nn.ModuleList):
    """
    Summary
    """
    def __init__(self):
        links = []
        super(EmptyModule, self).__init__(links)


# class EmptyChain(ChainList):
#     """
#     Summary
#     """
#     def __init__(self):
#         links = []
#         super(EmptyChain, self).__init__(*links)
