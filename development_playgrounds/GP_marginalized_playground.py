import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from brancher.variables import ProbabilisticModel

from brancher.stochastic_processes import GaussianProcess as GP
from brancher.stochastic_processes import SquaredExponentialCovariance as SquaredExponential
from brancher.stochastic_processes import WhiteNoiseCovariance as WhiteNoise
from brancher.stochastic_processes import PeriodicCovariance as Periodic
from brancher.stochastic_processes import ConstantMean
from brancher.variables import DeterministicVariable
from brancher.standard_variables import NormalVariable as Normal
from brancher.standard_variables import LogNormalVariable as LogNormal
from brancher.standard_variables import MultivariateNormalVariable as MultivariateNormal
from brancher import inference
from brancher.visualizations import plot_posterior
import brancher.functions as BF

num_datapoints = 60
x_range = np.linspace(-2, 2, num_datapoints)
x = DeterministicVariable(x_range, name="x")

# Model
length_scale = LogNormal(0., 0.3, name="length_scale")
noise_var = LogNormal(0., 0.3, name="noise_var")
freq = Normal(0.2, 0.2, name="freq")
mu = ConstantMean(0.)
cov = Periodic(frequency=freq, scale=length_scale, jitter=10**-5) + WhiteNoise(magnitude=noise_var)
f = GP(mu, cov, name="f")
y = f(x)
model = ProbabilisticModel([y])

# Observe data
noise_level = 0.2
data = np.sin(2*np.pi*0.3*x_range) + noise_level*np.random.normal(0., 1., (1, num_datapoints))
y.observe(data)

#Variational Model
Qlength_scale = LogNormal(-1, 0.2, name="length_scale", learnable=True)
Qnoise_var = LogNormal(-1, 0.2, name="noise_var", learnable=True)
Qfreq = Normal(0.2, 0.2, name="freq", learnable=True)
variational_model = ProbabilisticModel([Qlength_scale, Qnoise_var, Qfreq])
model.set_posterior_model(variational_model)

# Inference
inference.stochastic_variational_inference(model,
                                           number_iterations=4000,
                                           number_samples=30,
                                           optimizer='Adam',
                                           lr=0.005)
loss_list = model.diagnostics["loss curve"]
plt.plot(loss_list)
plt.show()

# Posterior plot
plot_posterior(model, variables=["length_scale", "noise_var", "freq"])
plt.show()