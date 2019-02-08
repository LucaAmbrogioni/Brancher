import numpy as np
import matplotlib.pyplot as plt

from brancher.stochastic_processes import GaussianProcess as GP
from brancher.stochastic_processes import SquaredExponentialCovariance as SquaredExponential
from brancher.stochastic_processes import ConstantMean
from brancher.variables import DeterministicVariable
from brancher.standard_variables import NormalVariable as Normal
from brancher.standard_variables import LogNormalVariable as LogNormal

x = DeterministicVariable(np.linspace(-1, 1, 20), name="x")

length_scale = LogNormal(loc=0, scale=0.5, name="length_scale")
mu = ConstantMean(0.)
cov = SquaredExponential(scale=length_scale)
f = GP(mu, cov, name="f")
y = Normal(f(x), 1., name="y")

print(y._get_sample(10))
