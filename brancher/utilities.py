"""
Utilities
---------
Module description
"""
import sys
from functools import reduce
from collections import abc
from collections.abc import Iterable

import numpy as np
import chainer
import chainer.functions as F

import torch as T


def to_tuple(obj):
    if isinstance(obj, Iterable):
        return tuple(obj)
    else:
        return tuple([obj])


def zip_dict(first_dict, second_dict):
    keys = set(first_dict.keys()).intersection(set(second_dict.keys()))
    return {k: to_tuple(first_dict[k]) + to_tuple(second_dict[k]) for k in keys}


def zip_dict_list(dict_list):
    if len(dict_list) == 0:
        return {}
    if len(dict_list) == 1:
        return dict_list[0]
    else:
        zipped_dict = zip_dict(dict_list[-1], dict_list[-2])
        new_dict_list = dict_list[:-2]
        new_dict_list.append(zipped_dict)
        return zip_dict_list(new_dict_list)


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


def flatten_set(st):
    flat_set = set([item for subset in st for item in subset])
    return flat_set


def join_dicts_list(dicts_list):
    if dicts_list:
        return reduce(lambda d1, d2: {**d1, **d2}, dicts_list)
    else:
        return {}


def join_sets_list(sets_list):
    if sets_list:
        return reduce(lambda d1, d2: d1.union(d2), sets_list)
    else:
        return set()


def sum_from_dim(tensor, dim_index):
    ''' replaced with torch.sum'''
    assert torch.is_tensor(tensor), 'object is not torch tensor' #TODO: later, either remove or replace with check for bracher tensor
    data_dim = len(tensor.shape)
    for dim in reversed(range(dim_index, data_dim)):
        tensor = tensor.sum(dim=dim)
    return tensor

def sum_from_dim_chainer(var, dim_index):
    data_dim = len(var.shape)
    for dim in reversed(range(dim_index, data_dim)):
        var = F.sum(var, axis=dim)
    return var


def sum_data_dimensions(var):
    return sum_from_dim(var, dim_index=2)

def partial_broadcast(*args):
    ''' replaced with torch.tensor.expand()'''
    assert all([T.is_tensor(ar) for ar in args]), 'at least 1 object is not torch tensor'
    shapes0, shapes1 = zip(*[(x.shape[0], x.shape[1]) for x in args])
    s0, s1 = np.max(shapes0), np.max(shapes1)
    return [x.expand((s0, s1) + x.shape[2:]) for x in args]

def partial_broadcast_chainer(*args):
    shapes0, shapes1 = zip(*[(x.shape[0], x.shape[1]) for x in args])
    s0, s1 = np.max(shapes0), np.max(shapes1)
    return [F.broadcast_to(x, shape=(s0, s1) + x.shape[2:]) for x in args]


def broadcast_and_squeeze(*args):
    ''' replaced with torch.tensor.reshape()'''
    assert all([T.is_tensor(ar) for ar in args]), 'at least 1 object is not torch tensor'
    if all([np.prod(val.shape[2:]) == 1 for val in args]):
        args = [val.reshape(shape=val.shape[:2] + tuple([1, 1])) for val in args] #TODO: Work in progress
    uniformed_values = uniform_shapes(*args)
    broadcasted_values = F.broadcast(*uniformed_values) #TODO: replace F.broadcast
    return broadcasted_values

def broadcast_and_squeeze_chainer(*args):
    if all([np.prod(val.shape[2:]) == 1 for val in args]):
        args = [F.reshape(val, shape=val.shape[:2] + tuple([1, 1])) for val in args] #TODO: Work in progress
    uniformed_values = uniform_shapes(*args)
    broadcasted_values = F.broadcast(*uniformed_values) #TODO: replace F.broadcast
    return broadcasted_values


def broadcast_parent_values(parents_values):
    ''' replaced with torch.tensor.reshape()'''
    keys_list, values_list = zip(*[(key, value) for key, value in parents_values.items()])
    broadcasted_values = partial_broadcast(*values_list)
    original_shapes = [val.shape for val in broadcasted_values]
    data_shapes = [s[2:] for s in original_shapes]
    number_samples, number_datapoints = original_shapes[0][0:2]
    newshapes = [tuple([number_samples * number_datapoints]) + s
                 for s in data_shapes]
    reshaped_values = [T.reshape(val, shape=s) for val, s in zip(broadcasted_values, newshapes)]
    return {key: value for key, value in zip(keys_list, reshaped_values)}, number_samples, number_datapoints

def broadcast_parent_values_chainer(parents_values):
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
    ''' replaced with torch.reshape()'''
    assert T.is_tensor(tensor), 'object is not torch tensor' #TODO: should be more checks here, what does it do?
    assert tensor.ndimension() == 4, 'ndim should be equal 4'
    dim1, dim2, dim_matrix, _ = tensor.shape
    diag_ind = list(range(dim_matrix))
    expanded_diag_ind = dim1*dim2*diag_ind
    axis12_ind = [a for a in range(dim1*dim2) for _ in range(dim_matrix)]
    reshaped_tensor = T.reshape(tensor, shape=(dim1*dim2, dim_matrix, dim_matrix))
    ind = (np.array(axis12_ind), np.array(expanded_diag_ind), np.array(expanded_diag_ind))
    subdiagonal = reshaped_tensor[ind]
    return T.reshape(subdiagonal, shape=(dim1, dim2, dim_matrix))

def get_diagonal_chainer(tensor):
    dim1, dim2, dim_matrix, _ = tensor.shape
    diag_ind = list(range(dim_matrix))
    expanded_diag_ind = dim1*dim2*diag_ind
    axis12_ind = [a for a in range(dim1*dim2) for _ in range(dim_matrix)]
    reshaped_tensor = F.reshape(tensor, shape = (dim1*dim2, dim_matrix, dim_matrix))
    ind = (np.array(axis12_ind), np.array(expanded_diag_ind), np.array(expanded_diag_ind))
    subdiagonal = reshaped_tensor[ind]
    return F.reshape(subdiagonal, shape=(dim1, dim2, dim_matrix))


def coerce_to_dtype(data, is_observed=False): #TODO: for Julia: Very important, add type checking: Tensor or Structure
    """Summary"""
    dtype = type(data)
    if dtype is chainer.Variable:
        result = data
    elif dtype is float or dtype is np.float32 or dtype is np.float64:
        result = chainer.Variable(data * np.ones(shape=(1, 1), dtype="float32"))
    elif dtype is int or dtype is np.int32 or dtype is np.int64:
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
        return result
    else:
        raise TypeError("Invalid input dtype {} - expected float, integer, np.ndarray, or chainer var.".format(dtype))

    if is_observed:
        result = F.expand_dims(result, axis=0)
        result_shape = result.shape
        if len(result_shape) == 2:
            result = F.reshape(result, shape=result_shape + tuple([1, 1]))
        elif len(result_shape) == 3:
            result = F.reshape(result, shape=result_shape + tuple([1]))
    else:
        result = F.expand_dims(F.expand_dims(result, axis=0), axis=1)
    return result


def tile_parameter(tensor, number_samples):
    ''' replaced with torch.tensor.repeat()'''
    assert T.is_tensor(tensor), 'object is not torch tensor'
    value_shape = tensor.shape
    if value_shape[0] == number_samples:
        return tensor
    elif value_shape[0] == 1:
        reps = tuple([number_samples] + [1] * len(value_shape[1:]))
        return tensor.repeat(repeats=reps)
    else:
        raise ValueError("The parameter cannot be broadcasted to the rerquired number of samples")

def tile_parameter_chainer(value, number_samples):
    value_shape = value.shape
    if value_shape[0] == number_samples:
        return value
    elif value_shape[0] == 1:
        reps = tuple([number_samples] + [1] * len(value_shape[1:]))
        return F.tile(value, reps=reps)
    else:
        raise ValueError("The parameter cannot be broadcasted to the rerquired number of samples")

def reformat_sampler_input(sample_input, number_samples):
    return {var: tile_parameter(coerce_to_dtype(value, is_observed=var.is_observed), number_samples=number_samples)
            for var, value in sample_input.items()}


def uniform_shapes(*args):
    ''' replaced with torch.unsqueeze()'''
    assert all([T.is_tensor(ar) for ar in args]), 'at least 1 object is not torch tensor'
    shapes = [ar.shape for ar in args]
    max_len = np.max([len(s) for s in shapes])
    return [T.unsqueeze(ar, dim=len(ar.shape)) if (len(ar.shape) == max_len-1) else ar
            for ar in args] #TODO: currently only works with unpacked input, should be flexible?

def uniform_shapes_chainer(*args):
    shapes = [ar.shape for ar in args]
    max_len = np.max([len(s) for s in shapes])
    return [F.expand_dims(ar, axis=len(ar.shape)) if (len(ar.shape) == max_len-1) else ar
            for ar in args]

def get_model_mapping(source_model, target_model):
    model_mapping = {}
    if isinstance(target_model, dict):
        target_variables = list(target_model.keys())
    else:
        target_variables = target_model._flatten()
    for p_var in target_variables:
        try:
            model_mapping.update({source_model.get_variable(p_var.name): p_var})
        except KeyError:
            pass
    return model_mapping


def reassign_samples(samples, model_mapping=(), source_model=(), target_model=()):
    out_sample = {}
    if model_mapping:
        pass
    elif source_model and target_model:
        model_mapping = get_model_mapping(source_model, target_model)
    else:
        raise ValueError("Either a model mapping or both source and target models have to be provided as input")
    for key, value in samples.items():
        try:
            out_sample.update({model_mapping[key]: value})
        except KeyError:
            pass
    return out_sample


def get_memory(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_memory(v, seen) for v in obj.values()])
        size += sum([get_memory(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_memory(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_memory(i, seen) for i in obj])
    return size


#TODO: Truncation material, to be cleaned up

def reject_samples(samples, model_statistics, truncation_rule): #TODO: Work in progress
    decision_variable = model_statistics(samples) #samples -> tensor
    sample_indices = [index for index, value in enumerate(decision_variable) if truncation_rule(value)]
    num_accepted_samples = len(sample_indices)
    if num_accepted_samples == 0:
        return None, 0, 0.1 #TODO: Improve
    else:
        remaining_samples = {var: value[sample_indices, :] for var, value in samples.items()}

        acceptance_probability = num_accepted_samples/float(decision_variable.shape[0])
        return remaining_samples, num_accepted_samples, acceptance_probability


def concatenate_samples(samples_list):
    ''' replaced with torch.cat'''
    if len(samples_list) == 1:
        return samples_list[0]
    else:
        paired_list = zip_dict_list(samples_list)
        samples = {var: T.cat(tensor_tuple, dim=0)
                   for var, tensor_tuple in paired_list.items()}
        return samples

