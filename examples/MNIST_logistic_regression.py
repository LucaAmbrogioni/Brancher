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
minibatch_size = 30
minibatch_indices = RandomIndices(dataset_size=dataset_size, batch_size=minibatch_size, name="indices", is_observed=True)
x = EmpiricalVariable(input_variable, indices=minibatch_indices, name="x", is_observed=True)
labels = EmpiricalVariable(output_labels, indices=minibatch_indices, name="labels", is_observed=True)

# Architecture parameters
weights = NormalVariable(np.zeros((number_output_classes, number_pixels)),
                         10*np.ones((number_output_classes, number_pixels)), "weights")

# Forward pass
final_activations = BF.matmul(weights, x)
k = CategoricalVariable(softmax_p=final_activations, name="k")

# Probabilistic model
model = ProbabilisticModel([k])

# Observations
k.observe(labels)

# Variational Model
Qweights = NormalVariable(np.zeros((number_output_classes, number_pixels)),
                          0.1*np.ones((number_output_classes, number_pixels)), "weights", learnable=True)
variational_model = ProbabilisticModel([Qweights])
model.set_posterior_model(variational_model)

# Inference
inference.stochastic_variational_inference(model,
                                           number_iterations=200,
                                           number_samples=30,
                                           optimizer=chainer.optimizers.Adam(0.005))

# Test accuracy
num_images = 500
test_size = len(test)
test_indices = RandomIndices(dataset_size=test_size, batch_size=1, name="test_indices", is_observed=True)
test_images = EmpiricalVariable(np.array([np.reshape(image[0], newshape=(number_pixels, 1)) for image in test]).astype("float32"),
                                indices=test_indices, name="x_test", is_observed=True)
test_labels = EmpiricalVariable(np.array([image[1]*np.ones((1, 1))
                                          for image in test]).astype("int32"), indices=test_indices, name="labels", is_observed=True)
test_model = ProbabilisticModel([test_images, test_labels])

s = 0
for _ in range(num_images):
    test_sample = test_model._get_sample(1)
    test_image, test_label = test_sample[test_images], test_sample[test_labels]
    model_output = np.reshape(np.mean(model._get_posterior_sample(10, input_values={x: test_image})[k].data, axis=0), newshape=(10,))
    s += 1 if int(np.argmax(model_output)) == int(test_label.data) else 0
print("Accuracy: {} %".format(100*s/float(num_images)))

#weight_map = variational_model._get_sample(1)[Qweights1].data[0, 0, 0, :]
#plt.imshow(np.reshape(weight_map, (28, 28)))
#plt.show()

plt.plot(model.diagnostics["loss curve"])
plt.show()