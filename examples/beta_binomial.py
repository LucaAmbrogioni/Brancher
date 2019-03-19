import matplotlib.pyplot as plt
import numpy as np

from brancher.variables import ProbabilisticModel
from brancher.standard_variables import BetaVariable, BinomialVariable
from brancher import inference

#Real model
number_samples = 1
p_real = 0.8
k_real = BinomialVariable(number_samples, p=p_real, name="k")

# betaNormal/Binomial model
p = BetaVariable(1., 1., "p")
k = BinomialVariable(number_samples, p=p, name="k")
model = ProbabilisticModel([k])

# Generate data
data = k_real._get_sample(number_samples=50)

# Observe data
k.observe(data[k_real][:, 0, :])

# Variational distribution
Qp = BetaVariable(1., 1., "p", learnable=True)
model.set_posterior_model(ProbabilisticModel([Qp]))

# Inference
inference.perform_inference(model,
                            number_iterations=3000,
                            number_samples=100,
                            lr=0.01,
                            optimizer='Adam')
loss_list = model.diagnostics["loss curve"]

#Plot results
plt.plot(loss_list)
plt.title("Loss (negative ELBO)")
plt.show()

from brancher.visualizations import plot_posterior

plot_posterior(model, variables=["p"])
plt.show()

