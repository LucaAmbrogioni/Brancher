"""
Utilities
---------
Module description
"""
from functools import reduce

import numpy as np
import chainer
import chainer.functions as F


def flatten_list(lst):
    flat_list = [item for sublist in lst for item in sublist]
    return flat_list


def join_dicts_list(dicts_list):
    if dicts_list:
        return reduce(lambda d1,d2: {**d1, **d2}, dicts_list)
    else:
        return {}


def join_sets_list(sets_list):
    if sets_list:
        return reduce(lambda d1, d2: d1.union(d2), sets_list)
    else:
        return set()


def sum_data_dimensions(var):
    data_dim = len(var.shape)
    for dim in reversed(range(2, data_dim)):
        var = F.sum(var, axis=dim)
    return var


def partial_broadcast(*args):
    shapes0, shapes1 = zip(*[(x.shape[0], x.shape[1]) for x in args])
    s0, s1 = np.max(shapes0), np.max(shapes1)
    return [F.broadcast_to(x, shape=(s0, s1) + x.shape[2:]) for x in args]


def broadcast_and_squeeze(*args):
    if all([np.prod(val.shape[2:]) == 1 for val in args]):
        args = [F.reshape(val, shape=val.shape[:2] + tuple([1, 1])) for val in args] #TODO: Work in progress
    broadcasted_values = F.broadcast(*args)
    return broadcasted_values


def get_diagonal(tensor):
    dim1, dim2, dim_matrix, _ = tensor.shape
    diag_ind = list(range(dim_matrix))
    expanded_diag_ind = dim1*dim2*diag_ind
    axis12_ind = [a for a in range(dim1*dim2) for _ in range(dim_matrix)]
    reshaped_tensor = F.reshape(tensor, shape = (dim1*dim2, dim_matrix, dim_matrix))
    ind = (np.array(axis12_ind), np.array(expanded_diag_ind), np.array(expanded_diag_ind))
    subdiagonal = reshaped_tensor[ind]
    return F.reshape(subdiagonal, shape=(dim1,dim2,dim_matrix))


def coerce_to_dtype(data, is_observed=False):
    """Summary"""
    dtype = type(data)
    if dtype is chainer.Variable:
        result = data
    elif np.issubdtype(dtype, float):
        result = chainer.Variable(data * np.ones(shape=(1, 1), dtype="float32")) #TODO
    elif np.issubdtype(dtype, int):
        result = chainer.Variable(data * np.ones(shape=(1, 1), dtype="int32"))
    elif dtype is np.ndarray:
        if data.dtype is np.dtype(np.float64):
            result = chainer.Variable(data.astype("float32"))
        elif data.dtype is np.dtype(np.int64):
            result = chainer.Variable(data.astype("int32"))
        else:
            result = chainer.Variable(data)
    else:
        raise TypeError("Invalid input dtype {} - expected float, integer, np.ndarray, or chainer var.".format(dtype))

    if is_observed:
        result = F.expand_dims(result, axis=0)
    else:
        result = F.expand_dims(F.expand_dims(result, axis=0), axis=1)
    return result
