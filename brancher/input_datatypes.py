"""
Input data types
---------
Module description
"""
import torch

class Tensor(torch.Tensor): #TODO: create a separate module (Values or InputTypes)
    """
    Tensor datatype: differentiable, inherited from torch.Tensor
    """

    # check torch.set_default_tensor_type(torch.DoubleTensor?)
    # torch.set_default_dtype
    # double, float, long, LongTensor

class Structure():
    """
    Structure datatype: non-differentiable, iter data type
    """