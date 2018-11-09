import chainer
import matplotlib.pyplot as plt
import numpy as np

from brancher.variables import DeterministicVariable, ProbabilisticModel
from brancher.standard_variables import NormalVariable, BinomialVariable, EmpiricalVariable, RandomIndices
from brancher import inference
import brancher.functions as BF

# Data
number_regressors = 2
dataset_size = 50
number_output_classes = 10
input_variable = np.concatenate((x1_input_variable, x2_input_variable), axis=0)
output_labels = np.concatenate((x1_labels, x2_labels), axis=0)

# Data sampling model
minibatch_size = 30
minibatch_indices = RandomIndices(dataset_size=dataset_size, batch_size=minibatch_size, name="indices", is_observed=True)
x = EmpiricalVariable(input_variable, indices=minibatch_indices, name="x", is_observed=True)
labels = EmpiricalVariable(output_labels, indices=minibatch_indices, name="labels", is_observed=True)

# Architecture parameters
number_hidden_units = 100
weights1 = NormalVariable(np.zeros((number_hidden_units, number_regressors)), 0.5*np.ones((1, number_regressors)), "weights1")
weights2 = NormalVariable(np.zeros((1, number_hidden_units)), 0.5*np.ones((1, number_regressors)), "weights2")

# Forward pass
hidden_units = BF.relu(BF.matmul(weights1, x))
final_activations = BF.matmul(weights2, hidden_units)
k = MultinomialVariable(1, logit_p=final_activations, name="k")

# Probabilistic model
model = ProbabilisticModel([k])

# Observations
k.observe(labels)

# Variational Model
Qweights1 = NormalVariable(np.zeros((number_hidden_units, number_regressors)),
                           np.ones((number_hidden_units, number_regressors)), "weights1", learnable=True)
Qweights2 = NormalVariable(np.zeros((number_output_classes, number_hidden_units)),
                           np.ones((number_output_classes, number_hidden_units)), "weights2", learnable=True)
variational_model = ProbabilisticModel([Qweights1, Qweights2])

# Inference
loss_list = inference.stochastic_variational_inference(model, variational_model,
                                                       number_iterations=200,
                                                       number_samples=100,
                                                       optimizer=chainer.optimizers.Adam(0.05))