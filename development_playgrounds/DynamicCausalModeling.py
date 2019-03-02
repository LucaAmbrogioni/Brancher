import matplotlib.pyplot as plt
import numpy as np

from brancher.variables import DeterministicVariable, RandomVariable, ProbabilisticModel
from brancher.standard_variables import NormalVariable, LogNormalVariable, BetaVariable
from brancher import inference
import brancher.functions as BF

# Probabilistic model
T = 2.
N = 100
dt = T/float(N)
time_range = np.linspace(0., T, N)
a = BetaVariable(1., 1., name="a")
b = BetaVariable(1., 1., name="b")
c = NormalVariable(0., 0.1, name="c")
d = NormalVariable(0., 0.1, name="d")
xi = LogNormalVariable(0.1, 0.1, name="xi")
chi = LogNormalVariable(0.1, 0.1, name="chi")
x_series = [NormalVariable(0., 1., name="x_0")]
y_series = [NormalVariable(0., 1., name="y_0")]
for n, t in enumerate(time_range):
    x_new_mean = (1-dt*a)*x_series[-1] + dt*c*y_series[-1]
    y_new_mean = (1-dt*b)*y_series[-1] + dt*d*x_series[-1]
    x_series += [NormalVariable(x_new_mean, np.sqrt(dt)*xi, name="x_{}".format(n+1))]
    y_series += [NormalVariable(x_new_mean, np.sqrt(dt)*chi, name="y_{}".format(n+1))]
dynamic_causal_model = ProbabilisticModel([x_series[-1], y_series[-1]])

# Run dynamics
sample = dynamic_causal_model.get_sample(number_samples=3)

# Observe
observable_data = sample[[x.name for x in x_series] + [y.name for y in y_series]]
dynamic_causal_model.observe(observable_data)

# Variational model

