import numpy as np
import matplotlib.pyplot as plt

from brancher.variables import ProbabilisticModel
from brancher.standard_variables import MultivariateNormalVariable

mean = np.zeros((2, 1))
chol_cov = np.array([[1., -1.],
                     [0., 4.]])

x = MultivariateNormalVariable(mean, chol_cov=chol_cov)

number_samples = 500
samples = x.get_sample(number_samples)
for sample in samples[x].data:
    plt.scatter(sample[0, 0, 0], sample[0, 1, 0], c="b")
plt.show()

mean = np.zeros((1, 2, 1))
diag_cov = np.ones((1, 2, 1))

y = MultivariateNormalVariable(mean, diag_cov=diag_cov)

number_samples = 500
samples = y.get_sample(number_samples)
for sample in samples[y].data:
    plt.scatter(sample[0, 0, 0], sample[0, 1, 0], c="b")
plt.show()