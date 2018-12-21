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
x = NormalVariable(mu=BF.sin(theta), sigma=0.3, name="x")
model = ProbabilisticModel([x, theta])

# Variational model
number_particles = 2
particle_locations = [DeterministicVariable(-1., name="location_1", learnable=True),
                      DeterministicVariable(1., name="location_2", learnable=True)]
Qtheta = EmpiricalVariable(dataset=BF.concat(particle_locations, axis=1), batch_size=1, weights= [0.1, 0.9], name="theta")
variational_model = ProbabilisticModel([Qtheta])

# Importance sampling distributions
voranoi_set = VoronoiSet(particle_locations)
variational_samplers = [TruncatedNormalVariable(mu=location, sigma=1., truncation_rule=voranoi_set(index), name="theta".format(index))
                        for index, location in enumerate(particle_locations)]

# Posterior model
model.set_posterior_model(model=variational_model, sampler=variational_samplers)

# Inference
inference.stochastic_variational_inference(model,
                                           inference_method=WVGD(),
                                           number_iterations=500,
                                           number_samples=50,
                                           optimizer=chainer.optimizers.Adam(0.025))
loss_list = model.diagnostics["loss curve"]

# Local variational models
#plt.plot(loss_list)

#samples = Qtheta.get_sample(50)
#print(samples)