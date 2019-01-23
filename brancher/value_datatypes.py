"""
Input data types
---------
Module description
"""
import torch
import numpy as np


class Tensor(torch.Tensor): #TODO: no copy, use ref to data
    """
    Tensor datatype: differentiable, inherited from torch.Tensor
    Input: scalar, ndarray or torch.tensor
    """

    # check torch.set_default_tensor_type(torch.DoubleTensor?)
    # torch.set_default_dtype
    # double, float, long, LongTensor


    def __new__(cls, data):
        dtype = type(data)
        if dtype in [float, int] or dtype.__base__ in [np.floating, np.signedinteger]:
            data = data * np.ones(shape=(1, 1))
        if dtype is torch.Tensor:
            data = data.type(torch.FloatTensor) # needs further checkinng
        return super().__new__(cls,data)

    def unsqueeze(self, dim):
        x = super().unsqueeze(dim)
        return Tensor(x)

    __module__ = 'brancher.value_datatypes'



        # if torch.is_tensor(data):
        #     super(Tensor, self).__init__()
        # else:
        #     dtype = type(data)
        #     if dtype is np.ndarray:
        #         #torch.from_numpy(data)
        #         super(Tensor, self).__init__()
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
    def __init__(self, data):
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


class Symbolic(Discrete, str):
    def __init__(self, data):
        assert(type(data) is str)
        self.value = data
        super(Symbolic, self).__init__(data)




# def coerce_to_dtype(data, is_observed=False):
#     def check_observed(result):
#         if is_observed:
#             result = torch.unsqueeze(result, dim=0)
#             result_shape = result.shape
#             if len(result_shape) == 2:
#                 result = result.view(size=result_shape + tuple([1, 1]))
#             elif len(result_shape) == 3:
#                 result = result.view(size=result_shape + tuple([1]))
#         else:
#             result = torch.unsqueeze(torch.unsqueeze(result, dim=0), dim=1)
#         return result
#
#     dtype = type(data)
#     if hasattr(dtype, '__module__') and dtype.__module__ == 'brancher.value_datatypes':
#         result = data
#     elif dtype in [float, int, np.ndarray, torch.Tensor] or dtype.__base__ in [np.floating, np.signedinteger]:
#         result = Tensor(data)
#     elif dtype is list:
#         result = DiscreteList(data)
#     elif dtype is set:
#         result = DiscreteSet(data)
#     elif dtype is tuple:
#         result = DiscreteTuple(data)
#     elif dtype is dict:
#         result = DiscreteDict(data)
#     elif dtype is str:
#         result = Symbolic(data)
#     else:
#         raise TypeError("Invalid input dtype {} - expected float, integer, np.ndarray, or torch var.".format(dtype))
#
#     return check_observed(result)