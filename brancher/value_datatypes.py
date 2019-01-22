"""
Input data types
---------
Module description
"""
import torch
import numpy as np
from collections import abc

class Tensor(torch.Tensor): #TODO: no copy, use ref to data
    """
    Tensor datatype: differentiable, inherited from torch.Tensor
    Input: scalar, ndarray or torch.tensor
    """

    # check torch.set_default_tensor_type(torch.DoubleTensor?)
    # torch.set_default_dtype
    # double, float, long, LongTensor

    def __init__(self, data): # no copying! only ref

        if torch.is_tensor(data):
            super(Tensor, self).__new__()
        else:
            dtype = type(data)
            if dtype is np.ndarray:
                #torch.from_numpy(data)
                super(Tensor, self).__init__()
           #  else:
           #      data = data * np.ones(shape=(1, 1))
           #      if dtype is float or dtype is np.float32 or dtype is np.float64:
           #          torch.from_numpy(data * np.ones(shape=(1, 1))) # dtype?
           #      elif dtype is int or dtype is np.int32 or dtype is np.int64:
           #          torch.from_numpy(data * np.ones(shape=(1, 1))) # dtype?
           # # super().new_tensor(data=data, dtype=dt)


class Discrete(abc.Iterable):
    """
    Discrete datatype: non-differentiable, iter data type: list/set/tuple; do not allow string; dictionary? grab items()?
    Input: list, set, tuple or dict
    """

    # make subclasses: list, set, tuple and dictionary inputs
    def __init__(self, data): # no copying! only ref
        self.value = data
        super(Discrete, self).__init__()

    def __iter__(self):
        for elem in self.value:
            yield elem


class DiscreteList(Discrete, list):
    def __init__(self, data): # no copying! only ref
        assert(type(data) is list)
        self.value = data
        super(DiscreteList, self).__init__(data)


class DiscreteSet(Discrete, set):
    def __init__(self, data):
        assert(type(data) is set)
        self.value = data
        super(DiscreteSet, self).__init__(data)


class DiscreteTuple(Discrete, tuple):
    def __init__(self, data):
        assert(type(data) is tuple)
        self.value = data
        super(DiscreteTuple, self).__init__(data)


class DiscreteDict(Discrete, dict):
    def __init__(self, data):
        assert(type(data) is dict)
        self.value = data
        super(DiscreteDict, self).__init__(data)

