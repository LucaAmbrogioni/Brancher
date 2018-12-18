"""
Inference
---------
Module description
"""
import warnings

import chainer
import chainer.functions as F
import numpy as np
from tqdm import tqdm

from brancher.optimizers import ProbabilisticOptimizer
from brancher.variables import DeterministicVariable, ProbabilisticModel


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
                                     input_values={}):
    """
    Summary

    Parameters
    ---------
    """
    joint_model.update_observed_submodel() #TODO: Probably not here
    posterior_model = joint_model.posterior_model
    joint_optimizer = ProbabilisticOptimizer(joint_model, optimizer)
    posterior_optimizer = ProbabilisticOptimizer(posterior_model, optimizer) #TODO: These things should not be here, maybe they should be inherited

    loss_list = []
    for iteration in tqdm(range(number_iterations)):
        loss = -joint_model.estimate_log_model_evidence(number_samples=number_samples,
                                                        method="ELBO", input_values=input_values, for_gradient=True)

        if np.isfinite(loss.data).all():
            posterior_optimizer.chain.cleargrads()
            joint_optimizer.chain.cleargrads()
            loss.backward()
            joint_optimizer.update()
            posterior_optimizer.update()
            loss_list.append(loss.data)
        else:
            warnings.warn("Numerical error, skipping sample")
    joint_model.diagnostics.update({"loss curve": np.array(loss_list)})
