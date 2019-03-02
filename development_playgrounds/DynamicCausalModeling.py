import matplotlib.pyplot as plt
import numpy as np

from brancher.variables import DeterministicVariable, RandomVariable, ProbabilisticModel
from brancher.standard_variables import NormalVariable, LogNormalVariable
from brancher import inference
from brancher.visualizations import plot_posterior

# Probabilistic model
T = 12.
N = 100
dt = T/float(N)
time_range = np.linspace(0., T, N)
a = LogNormalVariable(0., 1., name="a")
b = LogNormalVariable(0., 1., name="b")
c = NormalVariable(0., 1., name="c")
d = NormalVariable(0., 1., name="d")
xi = LogNormalVariable(0., 0.1, name="xi")
chi = LogNormalVariable(0., 0.1, name="chi")
x_series = [NormalVariable(0., 1., name="x_0")]
y_series = [NormalVariable(0., 1., name="y_0")]
for n, t in enumerate(time_range):
    x_new_mean = (1-dt*a)*x_series[-1] + dt*c*y_series[-1]
    y_new_mean = (1-dt*b)*y_series[-1] + dt*d*x_series[-1]
    x_series += [NormalVariable(x_new_mean, np.sqrt(dt)*xi, name="x_{}".format(n+1))]
    y_series += [NormalVariable(y_new_mean, np.sqrt(dt)*chi, name="y_{}".format(n+1))]
dynamic_causal_model = ProbabilisticModel([x_series[-1], y_series[-1]])

# Run dynamics
sample = dynamic_causal_model.get_sample(number_samples=1)

# Plot sample
time_series = sample[[x.name for x in x_series]].transpose().plot()
plt.show()

# Observe
observable_data = sample[[x.name for x in x_series] + [y.name for y in y_series]]
dynamic_causal_model.observe(observable_data)

# Variational model
Qa = LogNormalVariable(0., 0.5, name="a", learnable=True)
Qb = LogNormalVariable(0., 0.5, name="b", learnable=True)
Qc = NormalVariable(0., 0.1, name="c", learnable=True)
Qd = NormalVariable(0., 0.1, name="d", learnable=True)
Qxi = LogNormalVariable(0.1, 0.1, name="xi", learnable=True)
Qchi = LogNormalVariable(0.1, 0.1, name="chi", learnable=True)
variational_posterior = ProbabilisticModel([Qa, Qb, Qc, Qd, Qxi, Qchi])
dynamic_causal_model.set_posterior_model(variational_posterior)

# Inference #
inference.perform_inference(dynamic_causal_model,
                            number_iterations=1000,
                            number_samples=300,
                            optimizer='Adam',
                            lr=0.01)
loss_list = dynamic_causal_model.diagnostics["loss curve"]
plt.plot(loss_list)
plt.show()

# Plot posterior
plot_posterior(dynamic_causal_model, variables=["a", "b", "c", "d", "xi", "chi"])
plt.show()
