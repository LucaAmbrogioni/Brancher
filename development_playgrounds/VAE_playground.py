import torch
import torch.nn as nn
import torchvision

import matplotlib.pyplot as plt
import numpy as np

from brancher.variables import DeterministicVariable, ProbabilisticModel
from brancher.standard_variables import NormalVariable, EmpiricalVariable
from brancher import inference
import brancher.functions as BF

from brancher.config import device

# Data
image_size = 28*28
latent_size = 5

train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=None)
test = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=None)
dataset_size = len(train)
dataset = torch.Tensor(np.reshape(train.train_data.numpy(), newshape=(dataset_size, image_size, 1))).double().to(device)

# Neural architectures


## Encoder ##
class EncoderArchitecture(nn.Module):
    def __init__(self, image_size, latent_size, hidden_size=50):
        super(EncoderArchitecture, self).__init__()
        self.l1 = nn.Linear(image_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, latent_size) # Latent mean output
        self.l3 = nn.Linear(hidden_size, latent_size) # Latent log sd output
        self.softplus = nn.Softplus()

    def __call__(self, x):
        h = self.relu(self.l1(x[:, :, 0])) #TODO: to be fixed
        output_mean = self.l2(h)
        output_log_sd = self.l3(h)
        return {"mean": output_mean, "sd": self.softplus(output_log_sd) + 0.01}


## Decoder ##
class DecoderArchitecture(nn.Module):
    def __init__(self, latent_size, image_size, hidden_size=50):
        super(DecoderArchitecture, self).__init__()
        self.l1 = nn.Linear(latent_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, image_size) # Latent mean output
        self.l3 = nn.Linear(hidden_size, image_size) # Latent log sd output
        self.softplus = nn.Softplus()

    def __call__(self, x):
        h = self.relu(self.l1(x))
        output_mean = self.l2(h)
        output_log_sd = self.l3(h)
        return {"mean": output_mean, "sd": self.softplus(output_log_sd) + 0.01}


# Initialize encoder and decoders
encoder = BF.BrancherFunction(EncoderArchitecture(image_size=image_size, latent_size=latent_size))
decoder = BF.BrancherFunction(DecoderArchitecture(latent_size=latent_size, image_size=image_size))

# Generative model
z = NormalVariable(np.zeros((latent_size,)), np.ones((latent_size,)), name="z")
decoder_output = decoder(z)
x = NormalVariable(decoder_output["mean"], decoder_output["sd"], name="x")
model = ProbabilisticModel([x, z])

# Amortized variational distribution
Qx = EmpiricalVariable(dataset, batch_size=50, name="x", is_observed=True)
encoder_output = encoder(Qx)
Qz = NormalVariable(encoder_output["mean"], encoder_output["sd"], name="z")
model.set_posterior_model(ProbabilisticModel([Qx, Qz]))

# Joint-contrastive inference
inference.perform_inference(model,
                            number_iterations=5000,
                            number_samples=1,
                            optimizer="Adam",
                            lr=0.005)
loss_list = model.diagnostics["loss curve"]

#Plot results
plt.plot(loss_list)
plt.show()

sample = model.get_sample(1)
plt.imshow(np.reshape(sample["x"][0], newshape=(28, 28)))
plt.show()

sample = model.get_sample(1)
plt.imshow(np.reshape(sample["x"][0], newshape=(28, 28)))
plt.show()

sample = model.get_sample(1)
plt.imshow(np.reshape(sample["x"][0], newshape=(28, 28)))
plt.show()