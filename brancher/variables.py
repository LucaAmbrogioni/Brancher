"""
Variables
---------
Module description
"""
from abc import ABC, abstractmethod
import operator
import numbers
import collections

import chainer
import chainer.links as L
import chainer.functions as F
import numpy as np

from brancher.utilities import join_dicts_list, join_sets_list
from brancher.utilities import flatten_list
from brancher.utilities import partial_broadcast
from brancher.utilities import coerce_to_dtype
from brancher.utilities import broadcast_parent_values
from brancher.utilities import split_dict


class BrancherClass(ABC):
    """
    BrancherClass is the abstract superclass of all Brancher variables and models.
    """
    @abstractmethod
    def flatten(self):
        """
        Abstract method. It returs a list of all the variables contained in the model.
        """
        pass

    def get_variable(self, var_name):
        """
        It returns the variable in the model with the requested name.

        Args:
            var_name: String. Name  of the requested variable.

        Returns:
            brancher.Variable.

        """
        flat_list = self.flatten()
        try:
            return {var.name: var for var in flat_list}[var_name]
        except ValueError:
            raise ValueError("The variable {} is not present in the model".format(var_name))


class Variable(BrancherClass):
    """
    Variable is the abstract superclass of deterministic and random variables. Variables are the building blocks of
    all probabilistic models in Brancher.
    """
    @abstractmethod
    def calculate_log_probability(self, values, reevaluate):
        """
        Abstract method. It returns the log probability of the values given the model.

        Args:
            values: Dictionary(brancher.Variable: chainer.Variable). A dictionary having the brancher.variables of the
            model as keys and chainer.Variables as values. This dictionary has to provide values for all variables of
            the model except for the deterministic variables.

            reevaluate: Bool. If false it returns the output of the latest call. It avoid unnecessary computations when
            multiple children variables ask for the log probability of the same paternt variable.

        Returns:
            chainer.Variable. the log probability of the input values given the model.

        """
        pass

    @abstractmethod
    def get_sample(self, number_samples, resample):
        """
        Abstract method. It returns samples from the joint distribution specified by the model.

        Args:
            number_samples:

            resample:

        Returns:


        """
        pass

    @abstractmethod
    def reset(self):
        pass

    @property
    @abstractmethod
    def is_observed(self):
        pass

    def __str__(self):
        return self.name

    def _apply_operator(self, other, op):
        if isinstance(other, PartialLink):
            vars = other.vars
            vars.add(self)
            fn = lambda values: op(values[self], other.fn(values))
            links = other.links
        elif isinstance(other, Variable):
            vars = {self, other}
            fn = lambda values: op(values[self], values[other])
            links = set()
        elif isinstance(other, (numbers.Number, np.ndarray)):
            vars = {self}
            fn = lambda values: op(values[self], other)
            links = set()
        else:
            raise TypeError('') #TODO

        return PartialLink(vars=vars, fn=fn, links=links)

    def __add__(self, other):
        return self._apply_operator(other, operator.add)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self._apply_operator(other, operator.sub)

    def __rsub__(self, other):
        return -1*self.__sub__(other)

    def __mul__(self, other):
        return self._apply_operator(other, operator.mul)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return self._apply_operator(other, operator.truediv)

    def __rtruediv__(self, other):
        raise NotImplementedError

    def __pow__(self, other):
        return self._apply_operator(other, operator.pow)

    def __rpow__(self, other):
        raise NotImplementedError

    def __getitem__(self, key):
        if isinstance(key, collections.Iterable):
            variable_slice = (slice(None, None, None), *key)
        else:
            variable_slice = (slice(None, None, None), key)
        vars = {self}
        fn = lambda values: values[self][variable_slice]
        links = set()
        return PartialLink(vars=vars, fn=fn, links=links)

    def shape(self):
        vars = {self}
        fn = lambda values: values[self].shape
        links = set()
        return PartialLink(vars=vars, fn=fn, links=links)


class DeterministicVariable(Variable):
    """
    Summary

    Parameters
    ----------
    data : chainer.Variable, numeric, or np.ndarray
        Summary
    name : str
        Summary
    learnable : bool
        Summary
    """
    def __init__(self, data, name, learnable=False, is_observed=False):
        self._current_value = coerce_to_dtype(data, is_observed)
        self.name = name
        self._observed = is_observed
        self.parents = ()
        self.learnable = learnable
        if learnable:
            self.link = L.Bias(axis=1, shape=self._current_value.shape[1:])

    def calculate_log_probability(self, values, reevaluate=True):
        return 0.

    @property
    def value(self):
        assert self._current_value is not None
        if self.learnable:
            return self.link(self._current_value)
        return self._current_value

    @value.setter
    def value(self, val):
        self._current_value = coerce_to_dtype(val, is_observed=self.is_observed)

    @property
    def is_observed(self):
        return self._observed

    def get_sample(self, number_samples, resample=False, observed=False):
        if isinstance(self.value, chainer.Variable):
            value_shape = self.value.shape
            reps = tuple([number_samples] + [1]*len(value_shape[1:]))
            return {self: F.tile(self.value, reps=reps)}
        else:
            return {self: self.value} #TODO: This is for allowing discrete data, temporary?

    def reset(self):
        pass

    def flatten(self):
        return [self]


class RandomVariable(Variable):
    """
    Summary

    Parameters
    ----------
    distribution : brancher.Distribution
        Summary
    name : str
        Summary
    parents : tuple of brancher variables
        Summary
    link : callable
        Summary
    """
    def __init__(self, distribution, name, parents, link):
        self.name = name
        self.distribution = distribution
        self.link = link
        self.parents = parents
        self.samples = []

        self._evaluated = False
        self._observed = False
        self._observed_value = None
        self._current_value = None
        self.dataset = None
        self.has_random_dataset = False
        self.has_observed_value = False

    @property
    def value(self):
        if self._observed:
            return self._observed_value
        return self._current_value

    @value.setter
    def value(self, val):
        self._current_value = coerce_to_dtype(val)

    @property
    def is_observed(self):
        return self._observed

    def apply_link(self, parents_values):  #TODO: This is for allowing discrete data, temporary?
        cont_values, discrete_values = split_dict(parents_values,
                                                  condition=lambda key, val: isinstance(val, chainer.Variable))
        if cont_values:
            reshaped_dict, number_samples, number_datapoints = broadcast_parent_values(cont_values)
            reshaped_dict.update(discrete_values)
        else:
            reshaped_dict = discrete_values
        reshaped_output = self.link(reshaped_dict)
        output = {key: F.reshape(val, (number_samples, number_datapoints) + val.shape[1:])
                  if isinstance(val, chainer.Variable) else val
                  for key, val in reshaped_output.items()}
        return output

    def calculate_log_probability(self, input_values, reevaluate=True):
        """
        Summary
        """
        if self._evaluated and not reevaluate:
            return 0.
        if self in input_values:
            value = input_values[self]
        else:
            value = self.value

        self._evaluated = True
        deterministic_parents_values = {parent: parent.value for parent in self.parents
                                        if (type(parent) is DeterministicVariable)}
        parents_input_values = {parent: parent_input for parent, parent_input in input_values.items() if parent in self.parents}
        parents_values = {**parents_input_values, **deterministic_parents_values}
        parameters_dict = self.apply_link(parents_values)
        log_probability = self.distribution.calculate_log_probability(value, **parameters_dict)
        parents_log_probability = sum([parent.calculate_log_probability(input_values, reevaluate) for parent in self.parents])
        if self.is_observed:
            log_probability = F.sum(log_probability, axis=1, keepdims=True)
        if type(log_probability) is chainer.Variable and type(parents_log_probability) is chainer.Variable:
            log_probability, parents_log_probability = partial_broadcast(log_probability, parents_log_probability)
        return log_probability + parents_log_probability

    def get_sample(self, number_samples=1, resample=True, observed=False):
        """
        Summary
        """
        if self.samples and not resample:
            return {self: self.samples[-1]}
        if observed is False:
            var_to_sample = self
        else: #TODO: Some clean-up here
            if self.has_observed_value:
                return {self: self._observed_value}
            elif self.has_random_dataset:
                var_to_sample = self.dataset
            else:
                var_to_sample = self
        parents_samples_dict = join_dicts_list([parent.get_sample(number_samples, resample, observed)
                                                for parent in var_to_sample.parents])
        input_dict = {parent: parents_samples_dict[parent] for parent in var_to_sample.parents}
        parameters_dict = var_to_sample.apply_link(input_dict)
        sample = var_to_sample.distribution.get_sample(**parameters_dict, number_samples=number_samples)
        self.samples.append(sample)
        return {**parents_samples_dict, self: sample}

    def observe(self, data, random_indices=()):
        """
        Summary
        """
        if isinstance(data, RandomVariable): #TODO Work in progress: We need a small probabilistic model for keeping track of indices of datasets
            self.dataset = data #TODO: The sampler should construct the model from submodels
            self.has_random_dataset = True
        else:
            self._observed_value = coerce_to_dtype(data, is_observed=True)
            self.has_observed_value = True
        self._observed = True

    def unobserve(self):
        self._observed = False
        self.has_observed_value = False
        self.has_random_dataset = False
        self._observed_value = None
        self.dataset = None

    def reset(self):
        """
        Summary
        """
        self.samples = []
        self._evaluated = False
        self._current_value = None
        for parent in self.parents:
            parent.reset()

    def flatten(self):
        return flatten_list([parent.flatten() for parent in self.parents]) + [self]


class ProbabilisticModel(BrancherClass):
    """
    Summary

    Parameters
    ----------
    variables : tuple of brancher variables
        Summary
    """
    def __init__(self, variables):
        self.variables = self._validate_variables(variables)

    @staticmethod
    def _validate_variables(variables):
        for var in variables:
            if not isinstance(var, (DeterministicVariable, RandomVariable)):
                raise ValueError("Invalid input type: {}".format(type(var)))
        return variables

    @property
    def value(self):
        return tuple(var.value for var in self.variables)

    @value.setter
    def value(self, val):
        raise AttributeError("The value of a probabilistic model cannot be explicitly set.")

    @property
    def is_observed(self):
        return all([var.is_observed() for var in self.variables])

    def calculate_log_probability(self, rv_values):
        """
        Summary
        """
        log_probability = sum([var.calculate_log_probability(rv_values, reevaluate=False) for var in self.variables])
        self.reset()
        return log_probability

    def get_sample(self, number_samples, observed=False):
        """
        Summary
        """
        joint_sample = join_dicts_list([var.get_sample(number_samples=number_samples, resample=False, observed=observed)
                                        for var in self.variables])
        self.reset()
        return joint_sample

    def reset(self):
        """
        Summary
        """
        for variable in self.variables:
            variable.reset()

    def flatten(self):
        return flatten_list([var.flatten() for var in self.variables])


def var2link(var):
    if isinstance(var, Variable):
        vars = {var}
        fn = lambda values: values[var]
    elif isinstance(var, (numbers.Number, np.ndarray)):
        vars = {}
        fn = lambda values: var
    elif isinstance(var, tuple) and all([isinstance(v, (Variable, PartialLink)) for v in var]):
        vars = join_sets_list([{v} if isinstance(v, Variable) else v.vars for v in var])
        fn = lambda values: tuple([values[v] if isinstance(v, Variable) else v.fn(values) for v in var])
    else:
        return var
    return PartialLink(vars=vars, fn=fn, links=set())


class PartialLink(BrancherClass): #TODO: This should become "ProbabilisticProgram"

    def __init__(self, vars, fn, links):
        self.vars = vars
        self.fn = fn
        self.links = links

    def _apply_operator(self, other, op):
        other = var2link(other)
        return PartialLink(vars=self.vars.union(other.vars),
                           fn=lambda values: op(self.fn(values), other.fn(values)),
                           links=self.links.union(other.links))

    def __add__(self, other):
        return self._apply_operator(other, operator.add)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self._apply_operator(other, operator.sub)

    def __rsub__(self, other):
        return -1*self.__sub__(other)

    def __mul__(self, other):
        return self._apply_operator(other, operator.mul)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return self._apply_operator(other, operator.truediv)

    def __rtruediv__(self, other):
        raise NotImplementedError

    def __pow__(self, other):
        return self._apply_operator(other, operator.pow)

    def __rpow__(self, other):
        raise NotImplementedError

    def __getitem__(self, key):
        if isinstance(key, collections.Iterable):
            variable_slice = (slice(None, None, None), *key)
        else:
            variable_slice = (slice(None, None, None), key)
        vars = self.vars
        fn = lambda values: self.fn(values)[variable_slice]
        links = set()
        return PartialLink(vars=vars, fn=fn, links=links)

    def shape(self):
        vars = self.vars
        fn = lambda values: self.fn(values).shape
        links = set()
        return PartialLink(vars=vars, fn=fn, links=links)

    def flatten(self):
        return flatten_list([var.flatten() for var in self.vars]) + [self]
