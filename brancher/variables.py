"""
Variables
---------
Module description
"""
from abc import ABC, abstractmethod
import operator
import numbers
import collections
from collections.abc import Iterable

from brancher.modules import ParameterModule

import numpy as np
import pandas as pd
import torch

import warnings

from brancher.utilities import join_dicts_list, join_sets_list
from brancher.utilities import flatten_list
from brancher.utilities import partial_broadcast
from brancher.utilities import coerce_to_dtype
from brancher.utilities import broadcast_parent_values
from brancher.utilities import split_dict
from brancher.utilities import reformat_sampler_input
from brancher.utilities import tile_parameter
from brancher.utilities import get_model_mapping
from brancher.utilities import reassign_samples
from brancher.utilities import is_discrete, is_tensor
from brancher.utilities import concatenate_samples
from brancher.utilities import get_number_samples_and_datapoints

from brancher.pandas_interface import reformat_sample_to_pandas
from brancher.pandas_interface import reformat_model_summary
from brancher.pandas_interface import pandas_frame2dict
from brancher.pandas_interface import pandas_frame2value

from brancher.config import device

class BrancherClass(ABC):
    """
    BrancherClass is the abstract superclass of all Brancher variables and models.
    """
    @abstractmethod
    def _flatten(self):
        """
        Abstract method. It returs a list of all the variables contained in the model.
        """
        pass

    def flatten(self):
        return set(self._flatten())

    def get_variable(self, var_name):
        """
        It returns the variable in the model with the requested name.

        Args:
            var_name: String. Name  of the requested variable.

        Returns:
            torch.Tensor.

        """
        flat_list = self._flatten()
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
            torch.Tensor. the log probability of the input values given the model.

        """
        pass

    @abstractmethod
    def _get_sample(self, number_samples, resample, observed, input_values):
        """
        Abstract private method. It returns samples from the joint distribution specified by the model. If an input is provided
        it only samples the variables that are not contained in the input.

        Args:
            number_samples: Int.

            resample: Bool. If true it returns its previously stored sampled. It is used when multiple children variables
            ask for a sample to the same parent. In this case the resample variable is False since the children should be
            fed with the same value of the parent.

            observed: Bool. It specifies whether the samples should be interpreted frequentistically as samples from the
            observations of as Bayesian samples from the prior model. The first batch dimension is reserved to Bayesian
            samples while the second batch dimension is reserved to observation samples. Theerefore, this boolean
            changes the shape of the resulting sampled array.

            input_values: Dictionary(brancher.Variable, chainer.Variable). A dictionary having the brancher.variables of the
            model as keys and chainer.Variables as values. This dictionary has to provide values for all variables of
            the model that do not need to be sampled. Using an input allows to use a probabilistic model as a random
            function.

        Returns:
            Dictionary(brancher.Variable: torch.Tensor). A dictionary of samples from all the variables of the model

        """
        pass

    def get_sample(self, number_samples, input_values={}):
        reformatted_input_values = reformat_sampler_input(pandas_frame2dict(input_values),
                                                          number_samples=number_samples)
        raw_sample = {self: self._get_sample(number_samples, resample=False,
                                             observed=self.is_observed, input_values=reformatted_input_values)[self]}
        sample = reformat_sample_to_pandas(raw_sample, number_samples)
        self.reset()
        return sample

    @abstractmethod
    def reset(self):
        """
        Abstract method. It recursively resets the self._evalueted and self._current_value attributes of the variable and
        all downstream variables. It is used after sampling and evaluating the log probability of a model.

        Args: None.

        Returns: None.
        """
        pass

    @property
    @abstractmethod
    def is_observed(self):
        """
        Abstract property method. It returns True if the variable is observed and False otherwise.

        Args: None.

        Returns: Bool.
        """
        pass

    def __str__(self):
        """
        Method.

        Args: None

        Returns: String
        """
        return self.name

    def _apply_operator(self, other, op):
        """
        Method. It is used for performing symbolic operations between variables. It always returns a partialLink object
        that define a mathematical operation between variables. The vars attribute of the link is the set of variables
        that are used in the operation. The fn attribute is a lambda that specify the operation as a functions between the
        values of the variables in vars and a numeric output. This is required for defining the forward pass of the model.

        Args:
            other: PartialLink, RandomVariable, numeric or np.array.

            op: Binary operator.

        Returns: PartialLink
        """
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
            return other*self

        return PartialLink(vars=vars, fn=fn, links=links)

    def __neg__(self):
        return -1*self

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
        return self.__truediv__(other) ** (-1)

    def __pow__(self, other):
        return self._apply_operator(other, operator.pow)

    def __rpow__(self, other):
        raise NotImplementedError

    def __getitem__(self, key): #TODO: Work in progress
        if isinstance(key, str):
            variable_slice = key
        elif isinstance(key, Iterable):
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
    Deterministic variables are a subclass of random variables that always return the same value. The hyper-parameters of
    a probabilistic model are usually encoded as DeterministicVariables. When the user input a parameter as a Numeric value or
    an array, Brancher created a DeterministicVariable that store its value.

    Parameters
    ----------
    data : torch.Tensor, numeric, or np.ndarray. The value of the variable. It gets stored in the self.value
    attribute.

    name : String. The name of the variable.

    learnable : Bool. This boolean value specify if the value of the DeterministicVariable can be updated during traning.

    """
    def __init__(self, data, name, learnable=False, is_observed=False):
        self.name = name
        self._evaluated = False
        self._observed = is_observed
        self.parents = set()
        self.ancestors = set()
        self._type = "Deterministic"
        self.learnable = learnable
        self.link = None
        self._value = coerce_to_dtype(data, is_observed)
        if self.learnable:
            if not is_discrete(data):
                self._value = torch.nn.Parameter(coerce_to_dtype(data, is_observed), requires_grad=True)
                self.link = ParameterModule(self._value) # add to optimizer; opt checks links
            else:
                self.learnable = False
                warnings.warn('Currently discrete parameters are not learnable. Learnable set to False')


    def calculate_log_probability(self, values, reevaluate=True, for_gradient=False, normalized=True):
        """
        Method. It returns the log probability of the values given the model. This value is always 0 since the probability
        of a deterministic variable having its value is always 1.

        Args:
            values: Dictionary(brancher.Variable: chainer.Variable). A dictionary having the brancher.variables of the
            model as keys and chainer.Variables as values. This dictionary has to provide values for all variables of
            the model except for the deterministic variables.

            reevaluate: Bool. If false it returns the output of the latest call. It avoid unnecessary computations when
            multiple children variables ask for the log probability of the same paternt variable.

        Returns:
            torch.Tensor. The log probability of the input values given the model.
        """

        return torch.tensor(np.zeros((1,1))).float().to(device)

    @property
    def value(self):
        if self.learnable:
            return self.link()
        return self._value

    @property
    def is_observed(self):
        return self._observed

    def _get_sample(self, number_samples, resample=False, observed=False, input_values={}):
        if self in input_values:
            value = input_values[self]
        else:
            value = self.value
        if not is_discrete(value):
            return {self: tile_parameter(value, number_samples=number_samples)}
        else:
            return {self: value} #TODO: This is for allowing discrete data, temporary? (for Julia)

    def reset(self, recursive=False):
        pass

    def _flatten(self):
        return []


class RandomVariable(Variable):
    """
    Random variables are the main building blocks of probabilistic models.

    Parameters
    ----------
    distribution : brancher.Distribution
        The probability distribution of the random variable.
    name : str
        The name of the random variable.
    parents : tuple of brancher variables
        A tuple of brancher.Variables that are parents of the random variable.
    link : callable
        A function Dictionary(brancher.variable: torch.tensor) -> Dictionary(str: torch.tensor) that maps the values of all the parents
        to a dictionary of parameters of the probability distribution. It can also contains learnable layers and parameters.
    """
    def __init__(self, distribution, name, parents, link):
        self.name = name
        self.distribution = distribution
        self.link = link
        self.parents = parents
        self.ancestors = None
        self._type = "Random"
        self.samples = None

        self._evaluated = False
        self._observed = False # RandomVariable: observed value + link
        self._observed_value = None # need this?
        self.dataset = None
        self.has_random_dataset = False
        self.has_observed_value = False

    @property
    def value(self):
        """
        Method. It returns the value of the deterministic variable.

        Args:
            None

        Returns:
            torch.Tensor. The value of the deterministic variable.
        """
        if self._observed:
            return self._observed_value
        else:
            raise AttributeError('RandomVariable has to be observed to receive value.')

    @property
    def is_observed(self):
        return self._observed

    def _apply_link(self, parents_values):
        number_samples, number_datapoints = get_number_samples_and_datapoints(parents_values)
        cont_values, discrete_values = split_dict(parents_values,
                                                  condition=lambda key, val: not is_discrete(val))
        if cont_values:
            reshaped_dict = broadcast_parent_values(cont_values)
            reshaped_dict.update(discrete_values)
        else:
            reshaped_dict = discrete_values
        reshaped_output = self.link(reshaped_dict)
        output = {key: val.view(size=(number_samples, number_datapoints) + val.shape[1:])
                  if is_tensor(val) else val
                  for key, val in reshaped_output.items()}
        return output

    def calculate_log_probability(self, input_values, reevaluate=True, for_gradient=False,
                                  include_parents=True, normalized=True):
        """
        Method. It returns the log probability of the values given the model. This value is always 0 since the probability
        of a deterministic variable having its value is always 1.

        Args:
            values: Dictionary(brancher.Variable: chainer.Variable). A dictionary having the brancher.variables of the
            model as keys and chainer.Variables as values. This dictionary has to provide values for all variables of
            the model except for the deterministic variables.

            reevaluate: Bool. If false it returns the output of the latest call. It avoid unnecessary computations when
            multiple children variables ask for the log probability of the same paternt variable.

        Returns:
            torch.Tensor. The log probability of the input values given the model.

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
        parameters_dict = self._apply_link(parents_values)
        log_probability = self.distribution.calculate_log_probability(value, **parameters_dict)
        parents_log_probability = sum([parent.calculate_log_probability(input_values, reevaluate, for_gradient,
                                                                        normalized=normalized)
                                       for parent in self.parents])
        if self.is_observed:
            log_probability = log_probability.sum(dim=1, keepdim=True)
        if is_tensor(log_probability) and is_tensor(parents_log_probability):
            log_probability, parents_log_probability = partial_broadcast(log_probability, parents_log_probability)
        if include_parents:
            return log_probability + parents_log_probability
        else:
            return log_probability

    def _get_sample(self, number_samples=1, resample=True, observed=False, input_values={}):
        """
        Method. Used internally. It returns samples from the random variable and all its parents.

        Args:
            number_samples: . A dictionary having the brancher.variables of the
            model as keys and chainer.Variables as values. This dictionary has to provide values for all variables of
            the model except for the deterministic variables.

            resample: Bool. If false it returns the previously sampled values. It avoids that the parents of a variable
            are sampled multiple times.

            observed: Bool. It specifies if the sample should be formatted as observed data or Bayesian parameter.

            input_values: Dictionary(Variable: torch.Tensor).  dictionary of values of the parents. It is used for
            conditioning the sample on the (inputed) values of some of the parents.

        Returns:
            Dictionary(Variable: torch.Tensor). A dictionary of samples from the variable and all its parents.

        """
        if self.samples is not None and not resample:
            return {self: self.samples[-1]}
        if not observed:
            if self in input_values:
                return {self: input_values[self]}
            else:
                var_to_sample = self
        else:
            if self.has_observed_value:
                return {self: self._observed_value}
            elif self.has_random_dataset:
                var_to_sample = self.dataset
            else:
                var_to_sample = self
        parents_samples_dict = join_dicts_list([parent._get_sample(number_samples, resample, observed, input_values)
                                                for parent in var_to_sample.parents])
        input_dict = {parent: parents_samples_dict[parent] for parent in var_to_sample.parents}
        parameters_dict = var_to_sample._apply_link(input_dict)
        sample = var_to_sample.distribution.get_sample(**parameters_dict)
        self.samples = [sample] #TODO: to fix
        output_sample = {**parents_samples_dict, self: sample}
        return output_sample

    def observe(self, data):
        """
        Method. It assigns an observed value to a RandomVariable.

        Args:
            data: torch.Tensor, numeric, or np.ndarray. Input observed data.

        Returns:
            None

        """
        data = pandas_frame2value(data, self.name)
        if isinstance(data, RandomVariable):
            self.dataset = data
            self.has_random_dataset = True
        else:
            self._observed_value = coerce_to_dtype(data, is_observed=True)
            self.has_observed_value = True
        self._observed = True

    def unobserve(self):
        """
        Method. It marks a variable as not observed and it drops the dataset.

        Args:
            None

        Returns:
            None
        """
        self._observed = False
        self.has_observed_value = False
        self.has_random_dataset = False
        self._observed_value = None
        self.dataset = None

    def reset(self, recursive=True):
        """
        Method. It resets the evaluated flag of the variable and all its parents. Used after computing the
        log probability of a variable.
        """
        self.samples = None
        self._evaluated = False
        if recursive:
            for var in self.ancestors:
                var.samples = None
                var._evaluated = False

    def _flatten(self):
        variables = list(self.ancestors) + [self]
        return sorted(variables, key=lambda v: v.name)


class ProbabilisticModel(BrancherClass):
    """
    ProbabilisticModels are collections of Brancher variables.

    Parameters
    ----------
    variables: List(brancher.Variable). A list of random and deterministic variables.
    """
    def __init__(self, variables):
        self.variables = self._validate_variables(variables)
        self._set_summary()
        self.posterior_model = None
        self.posterior_sampler = None
        self.observed_submodel = None
        self.diagnostics = {}
        if not all([var.is_observed for var in self.variables]):
            self.update_observed_submodel()
        else:
            self.observed_submodel = self

    def __str__(self):
        """
        Method.

        Args: None

        Returns: String
        """
        return self.model_summary.__str__()

    @staticmethod
    def _validate_variables(variables):
        for var in variables:
            if not isinstance(var, (DeterministicVariable, RandomVariable, ProbabilisticModel)):
                raise ValueError("Invalid input type: {}".format(type(var)))
        return variables

    def _set_summary(self):
        feature_list = ["Distribution", "Parents", "Observed"]
        var_list = self.flatten()
        var_names = [var.name for var in var_list]
        summary_data = [[var._type, var.parents, var.is_observed]
                         for var in var_list]
        self._model_summary = reformat_model_summary(summary_data, var_names, feature_list)

    @property
    def model_summary(self):
        self._set_summary()
        return self._model_summary

    @property
    def is_observed(self):
        return all([var.is_observed for var in self._flatten()])

    def observe(self, data):
        if isinstance(data, pd.DataFrame):
            data = {var_name: pandas_frame2value(data, index=var_name) for var_name in data}
        if isinstance(data, dict):
            if all([isinstance(k, Variable) for k in data.keys()]):
                data_dict = data
            if all([isinstance(k, str) for k in data.keys()]):
                data_dict = {self.get_variable(name): value for name, value in data.items()}
        else:
            raise ValueError("The input data should be either a dictionary of values or a pandas dataframe")
        for var in data_dict:
            if isinstance(var, RandomVariable):
                var.observe(data_dict[var])

    def update_observed_submodel(self):
        """
        Method. Extract the sub-model of observed variables.
        """
        flattened_model = self._flatten()
        observed_variables = [var for var in flattened_model if var.is_observed]
        self.observed_submodel = ProbabilisticModel(observed_variables)

    def set_posterior_model(self, model, sampler=None): #TODO: Clean up code duplication
        self.posterior_model = PosteriorModel(posterior_model=model, joint_model=self)
        if sampler:
            if isinstance(sampler, ProbabilisticModel):
                self.posterior_sampler = PosteriorModel(sampler, joint_model=self)
            elif isinstance(sampler, Variable):
                self.posterior_sampler = PosteriorModel(ProbabilisticModel([sampler]), joint_model=self)
            elif isinstance(sampler, Iterable) and all([isinstance(subsampler, (ProbabilisticModel, Variable))
                                                        for subsampler in sampler]):
                self.posterior_sampler = [PosteriorModel(ProbabilisticModel([var]), joint_model=self)
                                          if isinstance(var, Variable) else PosteriorModel(var, joint_model=self)
                                          for var in sampler]
            else:
                raise ValueError("The sampler should be ither a probabilistic model, a brancher variable or an iterable of variables and/or models")

    def calculate_log_probability(self, rv_values, for_gradient=False, normalized=True):
        """
        Summary
        """
        log_probability = sum([var.calculate_log_probability(rv_values, reevaluate=False,
                                                             for_gradient=for_gradient,
                                                             normalized=normalized)
                               for var in self.variables])
        self.reset()
        return log_probability

    def _get_sample(self, number_samples, observed=False, input_values={}):
        """
        Summary
        """
        joint_sample = join_dicts_list([var._get_sample(number_samples=number_samples, resample=False,
                                                        observed=observed, input_values=input_values)
                                        for var in self.variables])
        joint_sample.update(input_values)
        self.reset()
        return joint_sample

    def get_sample(self, number_samples, input_values={}):
        reformatted_input_values = reformat_sampler_input(pandas_frame2dict(input_values),
                                                                            number_samples=number_samples)
        raw_sample = self._get_sample(number_samples, observed=False, input_values=reformatted_input_values)
        sample = reformat_sample_to_pandas(raw_sample, number_samples=number_samples)
        return sample

    def check_posterior_model(self):
        """
        Summary
        """
        if not self.posterior_model:
            raise AttributeError("The posterior model has not been initialized.")
#        elif self.posterior_model._is_trained is False:
#            raise AttributeError("The posterior model needs to be trained before sampling.")

    def _get_posterior_sample(self, number_samples, input_values={}):
        """
        Summary
        """
        self.check_posterior_model()
        posterior_sample = self.posterior_model._get_posterior_sample(number_samples=number_samples,
                                                                      input_values=input_values)
        sample = self._get_sample(number_samples, input_values=posterior_sample)
        return sample

    def get_posterior_sample(self, number_samples, input_values={}):
        reformatted_input_values = reformat_sampler_input(pandas_frame2dict(input_values),
                                                                            number_samples=number_samples)
        raw_sample = self._get_posterior_sample(number_samples, input_values=reformatted_input_values)
        sample = reformat_sample_to_pandas(raw_sample, number_samples=number_samples)
        return sample

    def get_p_and_q_log_probabilities(self, q_samples, q_model, empirical_samples={},
                                      for_gradient=False, normalized=True):  #TODO: Work in progress
        q_log_prob = q_model.calculate_log_probability(q_samples,
                                                       for_gradient=for_gradient, normalized=normalized)
        p_samples = reassign_samples(q_samples, source_model=q_model, target_model=self)
        p_samples.update(empirical_samples)
        p_log_prob = self.calculate_log_probability(p_samples, for_gradient=for_gradient, normalized=normalized)
        return q_log_prob, p_log_prob

    def get_importance_weights(self, q_samples, q_model, empirical_samples={},
                               for_gradient=False, give_normalization=False):
        if not empirical_samples:
            empirical_samples = self.observed_submodel._get_sample(1, observed=True)
        q_log_prob, p_log_prob = self.get_p_and_q_log_probabilities(q_samples=q_samples,
                                                                    q_model=q_model,
                                                                    empirical_samples=empirical_samples,
                                                                    for_gradient=for_gradient,
                                                                    normalized=False)
        log_weights = (p_log_prob - q_log_prob).detach().numpy()
        alpha = np.max(log_weights)
        norm_log_weights = log_weights - alpha
        weights = np.exp(norm_log_weights)
        norm = np.mean(weights)
        weights /= norm
        if not give_normalization:
            return weights
        else:
            return weights, np.log(norm) + alpha

    def estimate_log_model_evidence(self, number_samples, method="ELBO", input_values={}, for_gradient=False, posterior_model=()):
        if not posterior_model:
            self.check_posterior_model()
            posterior_model = self.posterior_model
        if method is "ELBO":
            empirical_samples = self.observed_submodel._get_sample(1, observed=True) #TODO Important!!: You need to correct for subsampling
            posterior_samples = posterior_model._get_sample(number_samples=number_samples,
                                                            observed=False, input_values=input_values)
            posterior_log_prob, joint_log_prob = self.get_p_and_q_log_probabilities(q_samples=posterior_samples,
                                                                                    empirical_samples=empirical_samples,
                                                                                    for_gradient=for_gradient,
                                                                                    q_model=posterior_model)
            log_model_evidence = torch.mean(joint_log_prob - posterior_log_prob)
            return log_model_evidence
        else:
            raise NotImplementedError("The requested estimation method is currently not implemented.")

    def reset(self):
        """
        Summary
        """
        for var in self.flatten():
            var.reset(recursive=False)

    def _flatten(self):
        variables = list(join_sets_list([var.ancestors.union({var}) for var in self.variables]))
        return sorted(variables, key=lambda v: v.name)


class PosteriorModel(ProbabilisticModel):
    """
    Summary

    Parameters
    ----------
    variables : tuple of brancher variables
        Summary
    """
    def __init__(self, posterior_model, joint_model):
        super().__init__(posterior_model.variables)
        self.posterior_model = None
        self.model_mapping = get_model_mapping(self, joint_model)

        self._is_trained = False

    def posterior_sample2joint_sample(self, posterior_sample):
        return reassign_samples(posterior_sample, self.model_mapping)

    def _get_posterior_sample(self, number_samples, observed=False, input_values={}):
        sample = self.posterior_sample2joint_sample(self._get_sample(number_samples, observed, input_values))
        sample.update(input_values)
        return sample


def var2link(var):
    if isinstance(var, Variable):
        vars = {var}
        fn = lambda values: values[var]
    elif isinstance(var, (numbers.Number, np.ndarray, torch.Tensor)):
        vars = set()
        fn = lambda values: var
    elif isinstance(var, (tuple, list)) and all([isinstance(v, (Variable, PartialLink)) for v in var]):
        vars = join_sets_list([{v} if isinstance(v, Variable) else v.vars for v in var])
        fn = lambda values: tuple([values[v] if isinstance(v, Variable) else v.fn(values) for v in var])
    else:
        return var
    return PartialLink(vars=vars, fn=fn, links=set())


class Ensemble(BrancherClass):

    def __init__(self, model_list, weights=None):
        #TODO: assert that all variables have the same name
        self.num_models = len(model_list)
        self.model_list = model_list
        if weights is None:
            self.weights = [1./self.num_models]*self.num_models
        else:
            self.weights = np.array(weights)

    def _get_sample(self, number_samples, observed=False, input_values={}):
        num_samples_list = np.random.multinomial(number_samples, self.weights)
        samples_list = [model._get_sample(n) for n, model in zip(num_samples_list, self.model_list)]
        named_sample_list = [{var.name: value for var, value in sample.items()} for sample in samples_list]
        named_sample = concatenate_samples(named_sample_list)
        sample = {self.model_list[0].get_variable(name): value for name, value in named_sample.items()}
        return sample

    def _flatten(self):
        return [model.flatten() for model in self.model_list]

    def get_sample(self, number_samples, input_values={}): #TODO: code duplication here
        reformatted_input_values = reformat_sampler_input(pandas_frame2dict(input_values),
                                                                            number_samples=number_samples)
        raw_sample = self._get_sample(number_samples, observed=False, input_values=reformatted_input_values)
        sample = reformat_sample_to_pandas(raw_sample, number_samples=number_samples)
        return sample



class PartialLink(BrancherClass):

    def __init__(self, vars, fn, links):
        self.vars = vars
        self.fn = fn
        self.links = links

    def _apply_operator(self, other, op):
        other = var2link(other)
        return PartialLink(vars=self.vars.union(other.vars),
                           fn=lambda values: op(self.fn(values), other.fn(values)),
                           links=self.links.union(other.links))

    def __neg__(self):
        return -1*self

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
        return self.__truediv__(other)**(-1)

    def __pow__(self, other):
        return self._apply_operator(other, operator.pow)

    def __rpow__(self, other):
        raise NotImplementedError

    def __getitem__(self, key):
        if isinstance(key, collections.Iterable) and all([isinstance(k, int) for k in key]):
            variable_slice = (slice(None, None, None), *key)
        elif isinstance(key, int):
            variable_slice = (slice(None, None, None), key)
        elif isinstance(key, collections.Hashable):
            variable_slice = key
        else:
            raise ValueError("The input to __getitem__ is neither numeric nor a hashabble key")

        vars = self.vars
        fn = lambda values: self.fn(values)[variable_slice] if is_tensor(self.fn(values)) \
            else self.fn(values)[key] #TODO: this should be a def not a lambda
        links = set()
        return PartialLink(vars=vars,
                           fn=fn,
                           links=self.links)

    def shape(self):
        vars = self.vars
        fn = lambda values: self.fn(values).shape
        links = set()
        return PartialLink(vars=vars,
                           fn=fn,
                           links=self.links)

    def _flatten(self):
        return flatten_list([var._flatten() for var in self.vars]) + [self]
