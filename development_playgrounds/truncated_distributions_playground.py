import chainer
import numpy as np
import matplotlib.pyplot as plt

from brancher.distributions import NormalDistribution, TruncatedDistribution

truncated_normal = TruncatedDistribution(base_distribution=NormalDistribution(), truncation_rule=lambda x: x > 0)


num_samples = 500
samples = truncated_normal.get_sample(num_samples, mu=chainer.Variable(0.5*np.ones((num_samples, 1, 1, 1))),
                                      sigma=chainer.Variable(np.ones((1, 1, 1, 1))))

p = truncated_normal.calculate_log_probability(samples, mu=chainer.Variable(np.zeros((num_samples, 1, 1, 1))),
                                                        sigma=chainer.Variable(np.ones((1, 1, 1, 1))))

print(p)

plt.hist(np.reshape(samples.data, newshape=(num_samples,)), 25)
plt.show()

