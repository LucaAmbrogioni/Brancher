import chainer
import matplotlib.pyplot as plt
import numpy as np

from brancher.variables import DeterministicVariable, ProbabilisticModel
from brancher.standard_variables import NormalVariable, CategoricalVariable, EmpiricalVariable, RandomIndices
from brancher import inference
import brancher.functions as BF

# Data
number_pixels = 28*28
number_output_classes = 10
train, test = chainer.datasets.get_mnist()
dataset_size = len(train)
input_variable = np.array([np.reshape(image[0], newshape=(number_pixels, 1)) for image in train]).astype("float32")
output_labels = np.array([image[1]*np.ones((1, 1)) for image in train]).astype("int32")

# Data sampling model
minibatch_size = 100
minibatch_indices = RandomIndices(dataset_size=dataset_size, batch_size=minibatch_size, name="indices", is_observed=True)
x = EmpiricalVariable(input_variable, indices=minibatch_indices, name="x", is_observed=True)
labels = EmpiricalVariable(output_labels, indices=minibatch_indices, name="labels", is_observed=True)

# Architecture parameters
number_hidden_units = 10
b1 = NormalVariable(np.zeros((number_hidden_units, 1)),
                    10*np.ones((number_hidden_units, 1)), "b1")
b2 = NormalVariable(np.zeros((number_output_classes, 1)),
                    10*np.ones((number_output_classes, 1)), "b2")
weights1 = NormalVariable(np.zeros((number_hidden_units, number_pixels)),
                          np.ones((number_hidden_units, number_pixels)), "weights1")
weights2 = NormalVariable(np.zeros((number_output_classes, number_hidden_units)),
                          np.ones((number_output_classes, number_hidden_units)), "weights2")

# Forward pass
hidden_units = BF.relu(BF.matmul(weights1, x) + b1)
final_activations = BF.matmul(weights2, hidden_units) + b2
k = CategoricalVariable(softmax_p=final_activations, name="k")

# Probabilistic model
model = ProbabilisticModel([k])

# Observations
k.observe(labels)

# Variational Model
Qb1 = NormalVariable(np.zeros((number_hidden_units, 1)),
                     0.01*np.ones((number_hidden_units, 1)), "b1")
Qb2 = NormalVariable(np.zeros((number_output_classes, 1)),
                     0.01 *np.ones((number_output_classes, 1)), "b2")
Qweights1 = NormalVariable(np.zeros((number_hidden_units, number_pixels)),
                           0.01 *np.ones((number_hidden_units, number_pixels)), "weights1", learnable=True)
Qweights2 = NormalVariable(np.zeros((number_output_classes, number_hidden_units)),
                           0.01 *np.ones((number_output_classes, number_hidden_units)), "weights2", learnable=True)
variational_model = ProbabilisticModel([Qb1, Qb2, Qweights1, Qweights2])

# Inference
loss_list = inference.stochastic_variational_inference(model, variational_model,
                                                       number_iterations=400,
                                                       number_samples=10,
                                                       optimizer=chainer.optimizers.Adam(0.02))

# Test accuracy
# test_batch_size = 100
# test_num_samples = 100
# test_size = len(test)
# input_variable_test = np.array([np.reshape(image[0], newshape=(number_pixels, 1)) for image in test]).astype("float32")
# output_labels_test = np.array([image[1]*np.ones((1,1)) for image in test]).astype("int32")
# minibatch_indices = RandomIndices(dataset_size=test_size, batch_size=test_batch_size, name="indices", is_observed=True)
# x_test = EmpiricalVariable(input_variable_test, indices=minibatch_indices, name="x", is_observed=True)
# labels_test = EmpiricalVariable(output_labels_test, indices=minibatch_indices, name="labels", is_observed=True)
# test_model = ProbabilisticModel([x_test, labels_test])
# k.unobserve()

weight_map = variational_model.get_sample(1)[Qweights1].data[0, 0, 0, :]
plt.imshow(np.reshape(weight_map, (28, 28)))
plt.show()

plt.plot(loss_list)
plt.show()