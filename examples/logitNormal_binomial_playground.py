import chainer
import chainer.functions as F
import matplotlib.pyplot as plt
import numpy as np

from brancher.variables import ProbabilisticModel
from brancher.standard_variables import LogitNormalVariable, BinomialVariable
from brancher import inference
import brancher.functions as BF

plt.close("all")

#Real model
number_samples = 1
p_real = 0.8
k_real = BinomialVariable(number_samples, p=p_real, name="k")

# LogitNormal/Binomial model
p = LogitNormalVariable(0.2, 2., "p")
k = BinomialVariable(number_samples, p=p, name="k")
model = ProbabilisticModel([k])

# Generate data
data = k_real.get_sample(number_samples=50)

# Observe data
k.observe(data[k_real][:,0,:])

# Variational distribution
Qp = LogitNormalVariable(0.2, 2., "p", learnable=True)
variational_model = ProbabilisticModel([Qp])

# Inference
loss_list = inference.stochastic_variational_inference(model, variational_model,
                                                       number_iterations=150,
                                                       number_samples=100,
                                                       optimizer=chainer.optimizers.Adam(0.05))

# Statistics
p_posterior_samples = variational_model.get_sample(2000)[Qp].data.flatten()

# Two subplots, unpack the axes array immediately
f, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(np.array(loss_list))
ax1.set_title("Convergence")
ax1.set_xlabel("Iteration")
ax2.hist(p_posterior_samples, 25)
ax2.axvline(x=p_real, lw=2, c="r")
ax2.set_title("Posterior samples (b)")
ax2.set_xlim(0,1)
plt.show()

