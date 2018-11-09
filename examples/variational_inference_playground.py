import chainer
import chainer.functions as F
import matplotlib.pyplot as plt
import numpy as np

from brancher.distributions import NormalDistribution
from brancher.links import GaussianLinearRegressionLink
from brancher.optimizers import ProbabilisticOptimizer
from brancher.variables import DeterministicVariable, RandomVariable, ProbabilisticModel
from brancher import inference

# Probabilistic model #
m0 = DeterministicVariable(data=0., name='m0')
v0 = DeterministicVariable(data=10., name='v0')
mu = RandomVariable(distribution=NormalDistribution(),
                    name='mu',
                    parents=(m0, v0),
                    link=lambda x: {"mean": x[m0], "var": x[v0]})
sigma = DeterministicVariable(data=0.1, name='sigma')
x = RandomVariable(distribution=NormalDistribution(),
                   name='x',
                   parents=(mu, sigma),
                   link=lambda x: {"mean": F.cos(x[mu]), "var": x[sigma]})
y = RandomVariable(distribution=NormalDistribution(),
                   name='y',
                   parents=(mu, sigma),
                   link=lambda x: {"mean": F.sin(x[mu]), "var": x[sigma]})
joint_var = ProbabilisticModel([x, y])

# Data
Rmu = DeterministicVariable(data=-1., name='mu')
Rsigma = DeterministicVariable(data=0.1, name='sigma')
Rx = RandomVariable(distribution=NormalDistribution(),
                    name='x',
                    parents=(Rmu, Rsigma),
                    link=lambda x: {"mean": F.cos(x[Rmu]), "var": x[Rsigma]})
Ry = RandomVariable(distribution=NormalDistribution(),
                    name='y',
                    parents=(Rmu, Rsigma),
                    link=lambda x: {"mean": F.sin(x[Rmu]), "var": x[Rsigma]})
Rjoint_var = ProbabilisticModel([Rx, Ry])
data = Rjoint_var.get_sample(number_samples=10)
x.observe(data[Rx][:,0,:])
y.observe(data[Ry][:,0,:])

print("The real mean is: {}".format(data[Rmu].data[0,0]))

# Variational model #
Qm0 = DeterministicVariable(data=0.,
                            name='m0',
                            learnable=True)
Qv0 = DeterministicVariable(data=2.,
                            name='v0',
                            learnable=True)
Qmu = RandomVariable(distribution=NormalDistribution(),
                     name='mu',
                     parents=(Qm0, Qv0),
                     link=lambda x: {"mean": x[Qm0], "var": F.exp(x[Qv0])})
variational_posterior = ProbabilisticModel([Qmu])

# Inference #
loss_list = inference.stochastic_variational_inference(joint_var, variational_posterior,
                                                       number_iterations=2000,
                                                       number_samples=500,
                                                       optimizer=chainer.optimizers.Adam(0.01)) #0,1

print("The estimated mean is: {} +- {}".format(Qmu.get_sample(1)[Qm0].data[0][0], np.sqrt(np.exp(Qmu.get_sample(1)[Qv0].data[0][0]))))

plt.plot(np.array(loss_list))
plt.title("Convergence")
plt.xlabel("Iteration")
plt.show()