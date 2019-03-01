import numpy as np
import matplotlib.pyplot as plt

from brancher.variables import ProbabilisticModel, Ensemble

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
from brancher.visualizations import plot_particles, plot_density
import brancher.functions as BF

num_datapoints = 50
x_range = np.linspace(0, 2, num_datapoints)
x = DeterministicVariable(x_range, name="x")

# Model
length_scale = LogNormal(0., 0.3, name="length_scale")
noise_var = LogNormal(0., 0.3, name="noise_var")
amplitude = LogNormal(0., 0.3, name="amplitude")
mu = ConstantMean(0.5)
cov = amplitude*SquaredExponential(scale=length_scale) + WhiteNoise(magnitude=noise_var)
f = GP(mu, cov, name="f")
y = f(x)
model = ProbabilisticModel([y])

# Observe data
noise_level = 0.3
f0 = 2.5
df = 0.
data = np.sin(2*np.pi*(f0 + df*x_range)*x_range) + noise_level*np.random.normal(0., 1., (1, num_datapoints))
y.observe(data)
plt.plot(x_range, data.flatten())
plt.show()


# Variational model
num_particles = 8
initial_locations = [(np.random.normal(0.8, 0.2), np.random.normal(0.8, 0.2), np.random.normal(0.8, 0.2))
                     for _ in range(num_particles)]
particles = [ProbabilisticModel([DeterministicVariable(location[0], name="length_scale", learnable=True),
                                 DeterministicVariable(location[1], name="noise_var", learnable=True),
                                 DeterministicVariable(location[2], name="amplitude", learnable=True)])
             for location in initial_locations]

# Importance sampling distributions
variational_samplers = [ProbabilisticModel([LogNormal(np.log(location[0]), 0.3, name="length_scale", learnable=True),
                                            LogNormal(np.log(location[1]), 0.3, name="noise_var", learnable=True),
                                            LogNormal(np.log(location[2]), 0.3, name="amplitude", learnable=True)])
                        for location in initial_locations]

# Inference
inference_method = WVGD(variational_samplers=variational_samplers,
                        particles=particles,
                        biased=False)
inference.perform_inference(model,
                            inference_method=inference_method,
                            number_iterations=2000,
                            number_samples=20,
                            optimizer="SGD",
                            lr=0.00025,
                            posterior_model=particles,
                            pretraining_iterations=0)
loss_list = model.diagnostics["loss curve"]
plt.plot(loss_list)
plt.show()

# Plot particles
plot_particles(particles,
               var_name="length_scale",
               var2_name="amplitude",
               c=inference_method.weights)
plt.show()

plot_particles(particles,
               var_name="length_scale",
               var2_name="noise_var",
               c=inference_method.weights)
plt.show()

plot_particles(particles,
               var_name="noise_var",
               var2_name="amplitude",
               c=inference_method.weights)
plt.show()

print(inference_method.weights)

final_ensemble = Ensemble(variational_samplers, inference_method.weights)
# Posterior plot
plot_density(final_ensemble, ["length_scale", "noise_var", "amplitude"], number_samples=3000)
plt.show()