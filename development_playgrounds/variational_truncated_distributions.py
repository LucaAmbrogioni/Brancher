import chainer
import matplotlib.pyplot as plt

from brancher.variables import ProbabilisticModel
from brancher import inference
from brancher.standard_variables import NormalVariable, LogNormalVariable, TruncatedNormalVariable
from brancher.visualizations import plot_posterior

# Model
a = TruncatedNormalVariable(mu=0., sigma=2., truncation_rule=lambda x: x > 0, name="a")
b = NormalVariable(mu=a, sigma=a**2, name="b")
model = ProbabilisticModel([a, b])

# Variational model
Qa = TruncatedNormalVariable(mu=1., sigma=0.25, truncation_rule=lambda x: x > 0.1, name="a", learnable=True)
variational_model = ProbabilisticModel([Qa])
model.set_posterior_model(variational_model)

# # Generate data
num_observations = 10
data = b.get_sample(number_samples=num_observations, input_values={a: 1.})

# Observe data
b.observe(data)

# Inference
inference.perform_inference(model,
                            number_iterations=500,
                            number_samples=50,
                            optimizer=chainer.optimizers.Adam(0.025))
loss_list = model.diagnostics["loss curve"]

plt.plot(loss_list)
plt.show()

plot_posterior(model, variables=["a", "b"])
plt.show()

