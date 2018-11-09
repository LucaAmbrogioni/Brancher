import chainer
import matplotlib.pyplot as plt
import numpy as np

from brancher.variables import DeterministicVariable, ProbabilisticModel
from brancher.standard_variables import NormalVariable, BinomialVariable
from brancher import inference
import brancher.functions as BF

# Data
number_regressors = 2
number_samples = 100
x1_input_variable = np.random.normal(1.5, 1., (int(number_samples/2), number_regressors, 1))
x1_labels = 0*np.ones((int(number_samples/2),1))
x2_input_variable = np.random.normal(-1.5, 1., (int(number_samples/2), number_regressors, 1))
x2_labels = 1*np.ones((int(number_samples/2),1))
input_variable = np.concatenate((x1_input_variable, x2_input_variable), axis=0)
labels = np.concatenate((x1_labels, x2_labels), axis=0)

# Probabilistic model
number_hidden_units = 10
w1 = NormalVariable(np.zeros((number_hidden_units, number_regressors)),
                    0.5*np.ones((number_hidden_units, number_regressors)), "w1")
w2 = NormalVariable(np.zeros((1, number_hidden_units)),
                    0.5*np.ones((1, number_hidden_units)), "w2")
x = DeterministicVariable(input_variable, "x", is_observed=True)
h = BF.tanh(BF.matmul(w1, x))
logit_p = BF.matmul(w2, h)
k = BinomialVariable(1, logit_p=logit_p, name="k")
model = ProbabilisticModel([k])

samples = model.get_sample(300)
model.calculate_log_probability(samples)

# Observations
k.observe(labels)

# Variational Model
Qw1 = NormalVariable(np.zeros((number_hidden_units, number_regressors)),
                     np.ones((number_hidden_units, number_regressors)), "w1", learnable=True)
Qw2 = NormalVariable(np.zeros((1, number_hidden_units)),
                     np.ones((1, number_hidden_units)), "w2", learnable=True)
variational_model = ProbabilisticModel([Qw1, Qw2])

# Inference
loss_list = inference.stochastic_variational_inference(model, variational_model,
                                                       number_iterations=100,
                                                       number_samples=100,
                                                       optimizer=chainer.optimizers.Adam(0.05))

# Statistics
from brancher.inference import get_variational_mapping, qsamples2psamples #TODO
posterior_samples = variational_model.get_sample(300)
x_range = np.linspace(-2, 2, 10)
y_range = np.linspace(-2, 2, 10)
input_var = np.array([[[x],[y]] for x in x_range for y in y_range])
x.value = input_var
var_map = get_variational_mapping(model, variational_model)
model.calculate_log_probability(qsamples2psamples(posterior_samples, var_map))

# Two subplots, unpack the axes array immediately
plt.plot(np.array(loss_list))
plt.show()
