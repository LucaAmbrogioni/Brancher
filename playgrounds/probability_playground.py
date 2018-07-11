from collections import OrderedDict
from itertools import product

import chainer
import chainer.functions as F
import matplotlib.pyplot as plt
import numpy as np

from brancher.distributions import NormalDistribution
from brancher.links import GaussianLinearRegressionLink
from brancher.optimizers import ProbabilisticOptimizer
from brancher.variables import DeterministicVariable, RandomVariable, ProbabilisticModel


DEFAULT_VAR_RANGE_PARAMS = {
    'start': -4,
    'stop': 4,
    'num': 30,
}


def calculate_log_likelihood(joint_variable, random_variables):
    shape = []
    var_ranges = OrderedDict()
    for var in random_variables:
        range_params = DEFAULT_VAR_RANGE_PARAMS
        shape.append(range_params['num'])
        try:
            var_ranges[var] = np.linspace(**range_params)
        except TypeError:
            raise ValueError("Invalid range parameters for random variable {}: {}".format(var.name, range_params))

    var_names = list(var_ranges.keys())
    range_combos = product(*list(var_ranges.values()))
    default_var = chainer.Variable(np.ones((1, 1), dtype="float32"))
    value_combos = [{var_names[i]: c * default_var for i, c in enumerate(combo)} for combo in range_combos]

    likelihood_ = []
    for values in value_combos:
        log_probability = joint_variable.calculate_log_probability(values)
        likelihood_.append(float(log_probability.data))

    likelihood = np.reshape(np.array(likelihood_), tuple(shape))

    return likelihood


m0 = DeterministicVariable(data=0.,
                           name='m0')
v0 = DeterministicVariable(data=1.,
                           name='v0')
mu = RandomVariable(distribution=NormalDistribution(),
                    name='mu',
                    parents=(m0, v0),
                    link=lambda x: {"mean": x[m0], "var": x[v0]})
sigma = DeterministicVariable(data=0.1,
                              name='sigma')
x = RandomVariable(distribution=NormalDistribution(),
                   name='x',
                   parents=(mu, sigma),
                   link=lambda x: {"mean": F.cos(x[mu]), "var": x[sigma]})
y = RandomVariable(distribution=NormalDistribution(),
                   name='y',
                   parents=(mu, sigma),
                   link=lambda x: {"mean": F.sin(x[mu]), "var": x[sigma]})

joint_var = ProbabilisticModel([x, y])
optimizer = ProbabilisticOptimizer()
optimizer.setup(joint_var)

a = joint_var.get_sample(number_samples=10)
p = joint_var.calculate_log_probability(a)

samples = [joint_var.get_sample(1) for _ in range(500)]

samples = [joint_var.get_sample(1) for _ in range(500)]
likelihood = calculate_log_likelihood(joint_var, (x, y, mu))

plt.imshow(np.transpose(np.sum(np.exp(likelihood), axis=2)), origin="lower", extent=[-4,4,-4,4])
plt.colorbar()
for sample in samples:
    plt.scatter(sample[x].data, sample[y].data, c = "r", alpha = 0.2)
    plt.xlim(-4,4)
    plt.ylim(-4,4)
plt.show()
