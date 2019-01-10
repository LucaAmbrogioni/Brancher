import numpy as np
import matplotlib.pyplot as plt
import chainer

from brancher.variables import DeterministicVariable, ProbabilisticModel
from brancher.particle_inference_tools import VoronoiSet
from brancher.standard_variables import EmpiricalVariable, NormalVariable, LogNormalVariable
from brancher import inference
from brancher.inference import WassersteinVariationalGradientDescent as WVGD
import brancher.functions as BF
from brancher.visualizations import ensemble_histogram

# Model
dimensionality = 1
theta = NormalVariable(mu=0., sigma=1., name="theta")
x = NormalVariable(mu=theta**2, sigma=0.4, name="x")
model = ProbabilisticModel([x, theta])

# Generate data
N = 4
theta_real = 0.5
x_real = NormalVariable(theta_real**2, 0.4, "x")
data = x_real._get_sample(number_samples=N)

# Observe data
x.observe(data[x_real][:, 0, :])

# Variational model
num_particles = 10
initial_locations = [np.random.normal(0., 1.)
                     for _ in range(num_particles)]
particles = [ProbabilisticModel([DeterministicVariable(p, name="theta", learnable=True)])
             for p in initial_locations]

# Importance sampling distributions
variational_samplers = [ProbabilisticModel([NormalVariable(mu=location, sigma=0.1,
                                                           name="theta", learnable=True)])
                        for location in initial_locations]

# Inference
inference_method = WVGD(variational_samplers=variational_samplers,
                        particles=particles,
                        biased=False)
inference.stochastic_variational_inference(model,
                                           inference_method=inference_method,
                                           number_iterations=250,
                                           number_samples=50,
                                           optimizer=chainer.optimizers.Adam(0.005),
                                           posterior_model=particles,
                                           pretraining_iterations=0)
loss_list = model.diagnostics["loss curve"]

# Local variational models
plt.plot(loss_list)
plt.show()

# Samples
print(inference_method.weights)
M = 2000
[sampler._get_sample(M) for sampler in inference_method.sampler_model]
samples = [sampler.get_sample(M) for sampler in inference_method.sampler_model]
ensemble_histogram(samples, "theta", bins=100)
# samples[0].hist(bins=30)
# plt.show()
# samples[1].hist(bins=30)
plt.show()

#samples = Qtheta.get_sample(50)
#print(samples)