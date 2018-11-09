import chainer
import matplotlib.pyplot as plt
import numpy as np

from brancher.variables import DeterministicVariable, ProbabilisticModel
from brancher.standard_variables import NormalVariable as Norm
from brancher.standard_variables import LogNormalVariable as LogNorm
from brancher import inference
import brancher.functions as BF

# Regressors
x_max = 1.
n = 100
x_range = np.reshape(np.linspace(-x_max, x_max, n), (n, 1, 1))
x1 = DeterministicVariable(np.sin(2*np.pi*2*x_range), name="x1", is_observed=True)
x2 = DeterministicVariable(x_range, name="x2", is_observed=True)

# Multivariate Regression
b = Norm(0., 1., name="b")
w1 = Norm(0., 1., name="w1")
w2 = Norm(0., 1., name="w2")
w12 = Norm(0., 1., name="w12")
nu = LogNorm(0.2, 0.5, name="nu")
mean = b + w1*x1 + w2*x2 + w12*x1*x2
y = Norm(mean, nu, name="y")
model = ProbabilisticModel([w1, w2, nu, y])

# Variational distributions
Qb = Norm(0., 1., name="b", learnable=True)
Qw1 = Norm(0., 1., name="w1", learnable=True)
Qw2 = Norm(0., 1., name="w2", learnable=True)
Qw12 = Norm(0., 1., name="w12", learnable=True)
Qnu = LogNorm(0.2, 0.5, name="nu", learnable=True)
variational_model = ProbabilisticModel([Qb, Qw1, Qw2, Qw12, Qnu])

# Generate data
ground_samples = model.get_sample(1)

# Observe data
data = np.reshape(ground_samples[y].data, newshape=(n, 1, 1))
y.observe(data)

# Inference
loss_list = inference.stochastic_variational_inference(model, variational_model,
                                                       number_iterations=200,
                                                       number_samples=100,
                                                       optimizer=chainer.optimizers.Adam(0.05))



# Variational Distribution
n_post_samples = 1000
post_samples = variational_model.get_sample(n_post_samples)
s_x1 = np.reshape(x1.value.data, newshape=(n,))
s_x2 = np.reshape(x2.value.data, newshape=(n,))
post_mean = 0.
for k in range(n_post_samples):
    s_b = float(post_samples[Qb].data[k,:])
    s_w1 = float(post_samples[Qw1].data[k,:])
    s_w2 = float(post_samples[Qw2].data[k,:])
    s_w12 = float(post_samples[Qw12].data[k,:])
    sample_function = s_b + s_w1*s_x1 + s_w2*s_x2 + s_w12*s_x1*s_x2
    post_mean += sample_function
    plt.plot(np.reshape(x_range, newshape=(n,)), sample_function, c="b", alpha=0.05)
post_mean /= float(n_post_samples)
plt.plot(np.reshape(x_range, newshape=(n,)), post_mean, c="k", lw=2, ls="--")
#plt.plot(loss_list)
plt.scatter(x_range, np.reshape(ground_samples[y].data, newshape=(n,)), c="k")
plt.xlim(-x_max, x_max)
plt.show()