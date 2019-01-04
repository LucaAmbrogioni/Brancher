import numpy as np
import matplotlib.pyplot as plt
import chainer

from brancher.variables import DeterministicVariable, ProbabilisticModel
from brancher.particle_inference_tools import VoronoiSet
from brancher.standard_variables import EmpiricalVariable, TruncatedNormalVariable, NormalVariable
from brancher import inference
from brancher.inference import WassersteinVariationalGradientDescent as WVGD
import brancher.functions as BF

# Model
dimensionality = 1
theta = NormalVariable(mu=0., sigma=1., name="theta")
x = NormalVariable(mu=theta**2 + 0.2*theta, sigma=0.4, name="x")
model = ProbabilisticModel([x, theta])

# Generate data
N = 10
theta_real = 0.5
x_real = NormalVariable(theta_real**2 + 0.2*theta_real, 0.4, "x")
data = x_real._get_sample(number_samples=N)

# Observe data
x.observe(data[x_real][:, 0, :])

# Variational model
number_particles = 2
#particle_locations = [DeterministicVariable(-1., name="theta_0", learnable=False),
#                      DeterministicVariable(1., name="theta_1", learnable=False)]
#Qtheta = EmpiricalVariable(dataset=BF.concat(particle_locations, axis=1), batch_size=1, weights=[0.1, 0.9],
#                           name="theta", learnable=False)
particle_1 = DeterministicVariable(-1., name="theta", learnable=True)
particle_2 = DeterministicVariable(1., name="theta", learnable=True)
particle_locations = [particle_1, particle_2]

# Importance sampling distributions
voranoi_set = VoronoiSet(particle_locations)
variational_samplers = [TruncatedNormalVariable(mu=-1., sigma=0.5, truncation_rule=lambda a: voranoi_set(a, 0),
                                                name="theta", learnable=True),
                        TruncatedNormalVariable(mu=1., sigma=0.5, truncation_rule=lambda a: voranoi_set(a, 1),
                                                name="theta", learnable=True)]

# Inference
inference.stochastic_variational_inference(model,
                                           inference_method=WVGD(),
                                           number_iterations=150,
                                           number_samples=50,
                                           optimizer=chainer.optimizers.Adam(0.005),
                                           posterior_model=particle_locations,
                                           sampler_model=variational_samplers) #TODO: You need the ensemble abstraction
loss_list = model.diagnostics["loss curve"]

# Local variational models
plt.plot(loss_list)
plt.show()

# Samples
M = 800
[sampler._get_sample(M) for sampler in variational_samplers]
samples = [sampler.get_sample(M) for sampler in variational_samplers]
samples[0].hist(bins=30)
plt.show()
samples[1].hist(bins=30)
plt.show()

#samples = Qtheta.get_sample(50)
#print(samples)