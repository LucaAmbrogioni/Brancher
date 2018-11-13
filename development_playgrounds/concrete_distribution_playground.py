import chainer
import matplotlib.pyplot as plt
import numpy as np

from brancher.variables import ProbabilisticModel
from brancher.standard_variables import ConcreteVariable, NormalVariable
from brancher import inference
import brancher.functions as BF

# Probabilistic Model
x = ConcreteVariable(tau=0.1, p=np.ones((2, 1))/2., name="x")
mu0 = -2
nu0 = 0.5
mu1 = 2
nu1 = 0.2
y = NormalVariable(x[0]*mu0 + x[1]*mu1, x[0]*nu0 + x[1]*nu1, "y")

samples = y.get_sample(1000)
plt.hist(samples[y].data.flatten(), 60)
print(y.calculate_log_probability(samples))
plt.title("Concrete mixture of Gaussians")
plt.show()