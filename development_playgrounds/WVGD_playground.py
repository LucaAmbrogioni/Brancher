import numpy as np
import matplotlib.pyplot as plt
import chainer

from brancher.variables import RootVariable, ProbabilisticModel
from brancher.particle_inference_tools import VoronoiSet
from brancher.standard_variables import EmpiricalVariable, NormalVariable, LogNormalVariable
from brancher import inference
from brancher.inference import WassersteinVariationalGradientDescent as WVGD
from brancher.visualizations import ensemble_histogram

# Model
dimensionality = 1
theta = NormalVariable(loc=0.5, scale=1., name="theta")
x = NormalVariable(loc=theta**2, scale=0.4, name="x")
model = ProbabilisticModel([x, theta])

# Generate data
N = 10
theta_real = 0.4
x_real = NormalVariable(theta_real**2, 0.4, "x")
data = x_real._get_sample(number_samples=N)

# Observe data
x.observe(data[x_real][:, 0, :])

# Variational model
num_particles = 4
initial_locations = [np.random.normal(0., 0.1)
                     for _ in range(num_particles)]
particles = [ProbabilisticModel([RootVariable(p, name="theta", learnable=True)])
             for p in initial_locations]

# Importance sampling distributions
variational_samplers = [ProbabilisticModel([NormalVariable(loc=location, scale=0.1,
                                                           name="theta", learnable=True)])
                        for location in initial_locations]

# Inference
inference_method = WVGD(variational_samplers=variational_samplers,
                        particles=particles,
                        biased=False,
                        number_post_samples=20000)
inference.perform_inference(model,
                            inference_method=inference_method,
                            number_iterations=200,
                            number_samples=50,
                            optimizer="Adam",
                            lr=0.0025,
                            posterior_model=particles,
                            pretraining_iterations=0)
loss_list = model.diagnostics["loss curve"]

# Local variational models
plt.plot(loss_list)
plt.show()

# Samples
print(inference_method.weights)
M = 5000
[sampler._get_sample(M) for sampler in inference_method.sampler_model]
samples = [sampler.get_sample(M) for sampler in inference_method.sampler_model]
ensemble_histogram(samples,
                   variable="theta",
                   weights=inference_method.weights,
                   bins=80)
plt.show()

print([p.get_sample(1) for p in particles])
print(initial_locations)

#samples = Qtheta.get_sample(50)
#print(samples)