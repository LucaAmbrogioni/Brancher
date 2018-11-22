"""
Utilities
---------
Module description
"""
from functools import reduce
from collections import abc

import numpy as np
import chainer
import chainer.functions as F


def split_dict(dic, condition):
    dict_1 = {}
    dict_2 = {}
    for key, val in dic.items():
        if condition(key, val):
            dict_1.update({key: val})
        else:
            dict_2.update({key: val})
    return dict_1, dict_2


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


def broadcast_parent_values(parents_values):
    keys_list, values_list = zip(*[(key, value) for key, value in parents_values.items()])
    broadcasted_values = partial_broadcast(*values_list)
    original_shapes = [val.shape for val in broadcasted_values]
    data_shapes = [s[2:] for s in original_shapes]
    number_samples, number_datapoints = original_shapes[0][0:2]
    newshapes = [tuple([number_samples * number_datapoints]) + s
                 for s in data_shapes]
    reshaped_values = [F.reshape(val, shape=s) for val, s in zip(broadcasted_values, newshapes)]
    return {key: value for key, value in zip(keys_list, reshaped_values)}, number_samples, number_datapoints


def get_diagonal(tensor):
    dim1, dim2, dim_matrix, _ = tensor.shape
    diag_ind = list(range(dim_matrix))
    expanded_diag_ind = dim1*dim2*diag_ind
    axis12_ind = [a for a in range(dim1*dim2) for _ in range(dim_matrix)]
    reshaped_tensor = F.reshape(tensor, shape = (dim1*dim2, dim_matrix, dim_matrix))
    ind = (np.array(axis12_ind), np.array(expanded_diag_ind), np.array(expanded_diag_ind))
    subdiagonal = reshaped_tensor[ind]
    return F.reshape(subdiagonal, shape=(dim1, dim2, dim_matrix))


def coerce_to_dtype(data, is_observed=False):
    """Summary"""
    dtype = type(data)
    if dtype is chainer.Variable:
        result = data
    elif np.issubdtype(dtype, float):
        result = chainer.Variable(data * np.ones(shape=(1, 1), dtype="float32"))
    elif np.issubdtype(dtype, int):
        result = chainer.Variable(data * np.ones(shape=(1, 1), dtype="int32"))
    elif dtype is np.ndarray:
        if data.dtype is np.dtype(np.float64):
            result = chainer.Variable(data.astype("float32"))
        elif data.dtype is np.dtype(np.int64):
            result = chainer.Variable(data.astype("int32"))
        else:
            result = chainer.Variable(data)
    elif issubclass(dtype, abc.Iterable):
        result = data  # TODO: This is for allowing discrete data, temporary?
        return result #TODO: This needs some clean up
    else:
        raise TypeError("Invalid input dtype {} - expected float, integer, np.ndarray, or chainer var.".format(dtype))

    if is_observed:
        result = F.expand_dims(result, axis=0)
    else:
        result = F.expand_dims(F.expand_dims(result, axis=0), axis=1)
    return result


def get_observed_model(probabilistic_model):
    """
    Summary

    Parameters
    ---------
    """
    flattened_model = probabilistic_model.flatten()
    observed_variables = [var for var in flattened_model if var.is_observed]
    return ProbabilisticModel(observed_variables)
