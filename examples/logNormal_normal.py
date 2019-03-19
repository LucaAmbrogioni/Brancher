import brancher.config as cfg
cfg.set_device("cpu")

import matplotlib.pyplot as plt
import numpy as np

from brancher.variables import ProbabilisticModel
from brancher.standard_variables import NormalVariable, LaplaceVariable, CauchyVariable, LogNormalVariable
from brancher import inference




# Real model
nu_real = 1.
mu_real = -2.
x_real = LaplaceVariable(mu_real, nu_real, "x_real")

# Normal model
nu = LogNormalVariable(0., 1., "nu")
mu = NormalVariable(0., 10., "mu")
x = LaplaceVariable(mu, nu, "x")
model = ProbabilisticModel([x])

# # Generate data
data = x_real._get_sample(number_samples=100)

# Observe data
x.observe(data[x_real][:, 0, :])

# Variational model
Qnu = LogNormalVariable(0., 1., "nu", learnable=True)
Qmu = NormalVariable(0., 1., "mu", learnable=True)
model.set_posterior_model(ProbabilisticModel([Qmu, Qnu]))

# Inference
inference.perform_inference(model,
                            number_iterations=3000,
                            number_samples=100,
                            optimizer='SGD',
                            lr=0.001)
loss_list = model.diagnostics["loss curve"]

plt.plot(loss_list)
plt.title("Loss (negative ELBO)")
plt.show()

from brancher.visualizations import plot_posterior

plot_posterior(model, variables=["mu", "nu", "x"])
plt.show()