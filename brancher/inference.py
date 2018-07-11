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
from brancher.variables import Variable, DeterministicVariable, RandomVariable, ProbabilisticModel


def get_variational_mapping(p,q):
    variational_mapping = {}
    for p_var in p.flatten():
        try:
            variational_mapping.update({q.get_variable(p_var.name): p_var})
        except KeyError:
            if p_var.is_observed or type(p_var) is DeterministicVariable:
                pass
            else:
                raise ValueError("The variable {} is not present in the variational distribution".format(p_var.name))
    return variational_mapping


def qsamples2psamples(qsamples, variational_mapping):
    p_samples = {}
    for key, value in qsamples.items():
        try:
            p_samples.update({variational_mapping[key]: value})
        except KeyError:
            pass
    return p_samples


def maximal_likelihood(random_variable, number_iterations, optimizer=chainer.optimizers.SGD(0.001)):
    """
    Summary

    Parameters
    ---------
    random_variable : brancher.Variable
    number_iterations : int
    optimizer : chainer.optimizers
    Summary
    """
    prob_optimizer = ProbabilisticOptimizer(optimizer)
    prob_optimizer.setup(random_variable)
    loss_list = []
    for iteration in tqdm(range(number_iterations)):
        loss = -F.sum(random_variable.calculate_log_probability({}))
        prob_optimizer.chain.cleargrads()
        loss.backward()
        prob_optimizer.optimizer.update()
        loss_list.append(loss.data)
    return loss_list


def stochastic_variational_inference(p, q, number_iterations, number_samples,
                                     optimizer=chainer.optimizers.Adam(0.001)):
    """
    Summary

    Parameters
    ---------
    p : brancher.Variable
    q : brancher.Variable
    optimizer : chainer.optimizers
    """
    variational_mapping = get_variational_mapping(p, q)
    prob_optimizer = ProbabilisticOptimizer(optimizer)
    prob_optimizer.setup(q)
    loss_list = []
    for iteration in tqdm(range(number_iterations)):
        q_samples = q.get_sample(number_samples)
        q_prob = q.calculate_log_probability(q_samples)
        p_prob = p.calculate_log_probability(qsamples2psamples(q_samples, variational_mapping))
        #p_prob, q_prob = F.broadcast(p_prob, q_prob)
        loss = -F.sum(p_prob - q_prob)
        if np.isfinite(loss.data).all():
            prob_optimizer.chain.cleargrads()
            loss.backward()
            prob_optimizer.optimizer.update()
            loss_list.append(loss.data)
        else:
            warnings.warn("Numerical error, skipping sample")
    return loss_list