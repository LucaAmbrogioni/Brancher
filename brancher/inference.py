"""
Inference
---------
Module description
"""
import warnings
from abc import ABC, abstractmethod
from collections.abc import Iterable

import chainer
import chainer.functions as F
import numpy as np
from tqdm import tqdm

from brancher.optimizers import ProbabilisticOptimizer
from brancher.variables import DeterministicVariable, Variable, ProbabilisticModel


# def maximal_likelihood(random_variable, number_iterations, optimizer=chainer.optimizers.SGD(0.001)):
#     """
#     Summary
#
#     Parameters
#     ---------
#     random_variable : brancher.Variable
#     number_iterations : int
#     optimizer : chainer.optimizers
#     Summary
#     """
#     prob_optimizer = ProbabilisticOptimizer(optimizer) #TODO: This function is not up to date
#     prob_optimizer.setup(random_variable)
#     loss_list = []
#     for iteration in tqdm(range(number_iterations)):
#         loss = -F.sum(random_variable.calculate_log_probability({}))
#         prob_optimizer.chain.cleargrads()
#         loss.backward()
#         prob_optimizer.optimizer.update()
#         loss_list.append(loss.data)
#     return loss_list


def stochastic_variational_inference(joint_model, number_iterations, number_samples,
                                     optimizer=chainer.optimizers.Adam(0.001),
                                     input_values={}, inference_method=None):
    """
    Summary

    Parameters
    ---------
    """
    if not inference_method:
        warnings.warn("The inference method was not specified, using the default reverse KL variational inference")
        inference_method = ReverseKL()

    joint_model.update_observed_submodel()
    posterior_model = joint_model.posterior_model
    sampler_model = joint_model.posterior_sampler

    optimizers_list = [ProbabilisticOptimizer(posterior_model, optimizer)]
    if inference_method.learnable_model:
        optimizers_list.append(ProbabilisticOptimizer(joint_model, optimizer))
    if inference_method.learnable_sampler:
        optimizers_list.append(ProbabilisticOptimizer(sampler_model, optimizer))

    loss_list = []

    inference_method.check_model_compatibility(joint_model, posterior_model, sampler_model)

    for iteration in tqdm(range(number_iterations)):
        loss = inference_method.compute_loss(joint_model, posterior_model, sampler_model, number_samples)

        if np.isfinite(loss.data).all():
            [opt.chain.cleargrads() for opt in optimizers_list]
            loss.backward()
            [opt.update() for opt in optimizers_list]
        else:
            warnings.warn("Numerical error, skipping sample")
        loss_list.append(loss.data)
    joint_model.diagnostics.update({"loss curve": np.array(loss_list)})


class InferenceMethod(ABC):

    #def __init__(self): #TODO: abstract attributes
    #   self.learnable_model = False
    #    self.needs_sampler = False
    #    self.learnable_sampler = False

    @abstractmethod
    def check_model_compatibility(self, joint_model, posterior_model, sampler_model):
        pass

    @abstractmethod
    def compute_loss(self, joint_model, posterior_model, sampler_model, number_samples, input_values):
        pass


class ReverseKL(InferenceMethod):

    def __init__(self):
        self.learnable_model = True
        self.needs_sampler = False
        self.learnable_sampler = False

    def check_model_compatibility(self, joint_model, posterior_model, sampler_model):
        pass #TODO: Check differentiability of the model

    def compute_loss(self, joint_model, posterior_model, sampler_model, number_samples, input_values={}):
        loss = -joint_model.estimate_log_model_evidence(number_samples=number_samples,
                                                        method="ELBO", input_values=input_values, for_gradient=True)
        return loss

class WassersteinVariationalGradientDescent(InferenceMethod): #TODO: Work in progress

    def __init__(self, cost_function=lambda x, y: F.sum((x-y)**2)):
        self.learnable_model = False #TODO: to implement later
        self.needs_sampler = True
        self.learnable_sampler = True
        self.cost_function = cost_function

    def check_model_compatibility(self, joint_model, posterior_model, sampler_model):
        assert isinstance(sampler_model, Iterable) and all([isinstance(subsampler, (Variable, ProbabilisticModel))
                                                            for subsampler in sampler_model]), "The Wasserstein Variational GD method require a list of variables or probabilistic models as sampler"
        # TODO: Check differentiability of the model

    def compute_loss(self, joint_model, posterior_model, sampler_model, number_samples, input_values={}): #TODO: Work in progress
        #particle_loss =
        samples = [subsampler._get_sample(number_samples)[:, 0, :]
                   for subsampler in sampler_model]
        #sampler_loss = sum([-joint_model.estimate_log_model_evidence(number_samples=number_samples, posterior_model=subsampler,
        #                                                             method="ELBO", input_values=input_values, for_gradient=True)
        #                    for subsampler in sampler_model])
        #return particle_loss + sampler_loss
        return sampler_loss #TODO: Work in progress


