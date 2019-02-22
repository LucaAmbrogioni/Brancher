"""
Inference
---------
Module description
"""
import warnings
from abc import ABC, abstractmethod
from collections.abc import Iterable

import numpy as np
from brancher.external.tqdm.tqdm import tqdm

import torch

from brancher.optimizers import ProbabilisticOptimizer
from brancher.variables import Variable, ProbabilisticModel
from brancher.transformations import truncate_model

from brancher.utilities import reassign_samples
from brancher.utilities import zip_dict
from brancher.utilities import sum_from_dim
from brancher.utilities import to_tensor


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
                                     optimizer='Adam', input_values={},
                                     inference_method=None,
                                     posterior_model=None, sampler_model=None,
                                     pretraining_iterations=0, **opt_params): #TODO: input values
    """
    Summary

    Parameters
    ---------
    """
    if not inference_method:
        warnings.warn("The inference method was not specified, using the default reverse KL variational inference")
        inference_method = ReverseKL()
    if not posterior_model:
        posterior_model = joint_model.posterior_model
    if not sampler_model: #TODO: clean up
        if not sampler_model:
            try:
                sampler_model = inference_method.sampler_model
            except AttributeError:
                try:
                    sampler_model = joint_model.posterior_sampler
                except AttributeError:
                    sampler_model = None

    joint_model.update_observed_submodel()

    def append_prob_optimizer(model, optimizer, **opt_params):
        prob_opt = ProbabilisticOptimizer(model, optimizer, **opt_params) # TODO: this should be better!!! handling models with no params
        if prob_opt.optimizer:
            optimizers_list.append(prob_opt)

    optimizers_list = []
    append_prob_optimizer(posterior_model, optimizer, **opt_params)
    if inference_method.learnable_model:
        append_prob_optimizer(joint_model, optimizer, **opt_params)
    if inference_method.learnable_sampler:
        append_prob_optimizer(sampler_model, optimizer, **opt_params)

    loss_list = []

    inference_method.check_model_compatibility(joint_model, posterior_model, sampler_model)

    for iteration in tqdm(range(number_iterations)):
        loss = inference_method.compute_loss(joint_model, posterior_model, sampler_model, number_samples)

        if torch.isfinite(loss.detach()).all().item(): #np.isfinite(loss.detach().numpy()).all(): #TODO: numpy()
            [opt.zero_grad() for opt in optimizers_list]
            loss.backward()
            optimizers_list[0].update()
            if iteration > pretraining_iterations:
                [opt.update() for opt in optimizers_list[1:]]
            loss_list.append(loss.detach().numpy())
        else:
            warnings.warn("Numerical error, skipping sample")
        loss_list.append(loss.cpu().detach().numpy())
    joint_model.diagnostics.update({"loss curve": np.array(loss_list)})

    inference_method.post_process(joint_model) #TODO: this could be implemented with a with block


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

    @abstractmethod
    def post_process(self, joint_model):
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

    def post_process(self, joint_model):
        pass

class WassersteinVariationalGradientDescent(InferenceMethod): #TODO: Work in progress

    def __init__(self, variational_samplers, particles,
                 cost_function=None,
                 deviation_statistics=None,
                 biased=False,
                 number_post_samples=8000): #TODO: Work in progress
        self.learnable_model = False #TODO: to implement later
        self.needs_sampler = True
        self.learnable_sampler = True
        self.biased = biased
        self.number_post_samples = number_post_samples
        if cost_function:
            self.cost_function = cost_function
        else:
            self.cost_function = lambda x, y: sum_from_dim((to_tensor(x) - to_tensor(y)) **2, dim_index=1)
        if deviation_statistics:
            self.deviation_statistics = deviation_statistics
        else:
            self.deviation_statistics = lambda lst: sum(lst)

        def model_statistics(dic):
            num_samples = list(dic.values())[0].shape[0]
            reassigned_particles = [reassign_samples(p._get_sample(num_samples), source_model=p, target_model=dic)
                                    for p in particles]

            statistics = [self.deviation_statistics([self.cost_function(value_pair[0], value_pair[1]).detach().numpy() #TODO: same as above + GPU
                                                     for var, value_pair in zip_dict(dic, p).items()])
                          for p in reassigned_particles]
            return np.array(statistics).transpose()

        truncation_rules = [lambda a, idx=index: True if (idx == np.argmin(a)) else False
                            for index in range(len(particles))]

        self.sampler_model = [truncate_model(model=sampler,
                                             truncation_rule=rule,
                                             model_statistics=model_statistics)
                              for sampler, rule in zip(variational_samplers, truncation_rules)]

    def check_model_compatibility(self, joint_model, posterior_model, sampler_model):
        assert isinstance(sampler_model, Iterable) and all([isinstance(subsampler, (Variable, ProbabilisticModel))
                                                            for subsampler in sampler_model]), "The Wasserstein Variational GD method require a list of variables or probabilistic models as sampler"
        # TODO: Check differentiability of the model

    def compute_loss(self, joint_model, posterior_model, sampler_model, number_samples, input_values={}):
        sampler_loss = sum([-joint_model.estimate_log_model_evidence(number_samples=number_samples, posterior_model=subsampler,
                                                                      method="ELBO", input_values=input_values, for_gradient=True)
                             for subsampler in sampler_model])
        particle_loss = self.get_particle_loss(joint_model, posterior_model, sampler_model, number_samples,
                                               input_values)
        return sampler_loss + particle_loss

    def get_particle_loss(self, joint_model, particle_list, sampler_model, number_samples, input_values):
        samples_list = [sampler._get_sample(number_samples, input_values=input_values)
                         for sampler in sampler_model]
        if self.biased:
            importance_weights = [1./number_samples for _ in sampler_model]
        else:
            importance_weights = [joint_model.get_importance_weights(q_samples=samples,
                                                                       q_model=sampler,
                                                                       for_gradient=False).flatten()
                                  for samples, sampler in zip(samples_list, sampler_model)]
        reassigned_samples_list = [reassign_samples(samples, source_model=sampler, target_model=particle)
                                   for samples, sampler, particle in zip(samples_list, sampler_model, particle_list)]
        pair_list = [zip_dict(particle._get_sample(1), samples)
                     for particle, samples in zip(particle_list, reassigned_samples_list)]

        particle_loss = sum([torch.sum(to_tensor(w)*self.deviation_statistics([self.cost_function(value_pair[0], value_pair[1].detach().numpy()) #TODO: numpy()
                                                                for var, value_pair in particle.items()]))
                             for particle, w in zip(pair_list, importance_weights)])
        return particle_loss

    def post_process(self, joint_model): #TODO: Work in progress
        sample_list = [sampler._get_sample(self.number_post_samples)
                        for sampler in self.sampler_model]
        self.weights = []
        for sampler, s in zip(self.sampler_model, sample_list):
            a = sampler.get_acceptance_probability(number_samples=self.number_post_samples)
            _, Z = joint_model.get_importance_weights(q_samples=s,
                                                      q_model=sampler,
                                                      for_gradient=False,
                                                      give_normalization=True)
            self.weights.append(a*Z)
        self.weights /= np.sum(self.weights)


