import numpy as np

from brancher.variables import DeterministicVariable
from brancher.particle_inference_tools import VoronoiSet
from brancher.standard_variables import EmpiricalVariable, TruncatedNormalVariable
import brancher.functions as BF

# Model
dimensionality = 1

# Variational model
number_particles = 2
particle_locations = [DeterministicVariable(-1., name="location_1", learnable=True),
                      DeterministicVariable(1., name="location_2", learnable=True)]
Qtheta = EmpiricalVariable(dataset=BF.concat(particle_locations, axis=1), batch_size=1, name="theta")

# Importance sampling distributions
voranoi_set = VoronoiSet(particle_locations)
[TruncatedNormalVariable(mu=location, sigma=1., truncation_rule=voranoi_set(index), name="sampler_{}".format(index))
 for index, location in enumerate(particle_locations)]

# Posterior model

# Inference

# Local variational models
sample = Qtheta.get_sample(30)
print(sample)
print(voranoi_set(0)(-0.1))
pass