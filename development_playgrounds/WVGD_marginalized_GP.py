import numpy as np
import matplotlib.pyplot as plt

from brancher.variables import ProbabilisticModel

from brancher.stochastic_processes import GaussianProcess as GP
from brancher.stochastic_processes import SquaredExponentialCovariance as SquaredExponential
from brancher.stochastic_processes import WhiteNoiseCovariance as WhiteNoise
from brancher.stochastic_processes import HarmonicCovariance as Harmonic
from brancher.stochastic_processes import ConstantMean
from brancher.variables import DeterministicVariable
from brancher.standard_variables import NormalVariable as Normal
from brancher.standard_variables import LogNormalVariable as LogNormal
from brancher.inference import WassersteinVariationalGradientDescent as WVGD
from brancher import inference
from brancher.visualizations import plot_particles
import brancher.functions as BF

num_datapoints = 35
x_range = np.linspace(-1, 1, num_datapoints)
x = DeterministicVariable(x_range, name="x")

# Model
length_scale = LogNormal(0., 0.3, name="length_scale")
noise_var = LogNormal(0., 0.3, name="noise_var")
freq = Normal(0.5, 0.5, name="freq")
mu = ConstantMean(0.5)
cov = SquaredExponential(scale=length_scale)*Harmonic(frequency=freq) + WhiteNoise(magnitude=noise_var)
f = GP(mu, cov, name="f")
y = f(x)
model = ProbabilisticModel([y])

# Observe data
noise_level = 0.4
f = 1.5
data = np.sin(2*np.pi*f*x_range) + noise_level*np.random.normal(0., 1., (1, num_datapoints))
y.observe(data)


# Variational model
num_particles = 8
initial_locations = [(np.random.normal(0.8, 0.2), np.random.normal(0.8, 0.2), np.random.normal(0.5, 0.2))
                     for _ in range(num_particles)]
particles = [ProbabilisticModel([DeterministicVariable(location[0], name="length_scale", learnable=True),
                                 DeterministicVariable(location[1], name="noise_var", learnable=True),
                                 DeterministicVariable(location[2], name="freq", learnable=True)])
             for location in initial_locations]

# Importance sampling distributions
variational_samplers = [ProbabilisticModel([LogNormal(np.log(location[0]), 0.3, name="length_scale", learnable=True),
                                            LogNormal(np.log(location[1]), 0.3, name="noise_var", learnable=True),
                                            Normal(location[2], 0.3, name="freq", learnable=True)
                                            ])
                        for location in initial_locations]

# Inference
inference_method = WVGD(variational_samplers=variational_samplers,
                        particles=particles,
                        biased=False)
inference.perform_inference(model,
                            inference_method=inference_method,
                            number_iterations=1500,
                            number_samples=20,
                            optimizer="SGD",
                            lr=0.001,
                            posterior_model=particles,
                            pretraining_iterations=0)
loss_list = model.diagnostics["loss curve"]
plt.plot(loss_list)
plt.show()

# Plot particles
plot_particles(particles,
               var_name="length_scale",
               var2_name="freq",
               c=inference_method.weights)
plt.show()

plot_particles(particles,
               var_name="length_scale",
               var2_name="noise_var",
               c=inference_method.weights)
plt.show()

plot_particles(particles,
               var_name="noise_var",
               var2_name="freq",
               c=inference_method.weights)
plt.show()

print(inference_method.weights)