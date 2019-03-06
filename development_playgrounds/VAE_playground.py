import torch
import torch.nn as nn
import torchvision

import matplotlib.pyplot as plt
import numpy as np

from brancher.variables import DeterministicVariable, ProbabilisticModel
from brancher.standard_variables import NormalVariable, EmpiricalVariable, BinomialVariable, DeterministicNode
from brancher import inference
import brancher.functions as BF

from brancher.config import device

# Data
image_size = 28*28
latent_size = 2

train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=None)
test = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=None)
dataset_size = len(train)
#dataset = torch.Tensor(np.reshape(train.train_data.numpy(), newshape=(dataset_size, image_size, 1))).double().to(device)
dataset = np.reshape(train.train_data.numpy(), newshape=(dataset_size, image_size, 1))
data_mean = np.mean(dataset)
dataset = (dataset > data_mean).astype("int32")

## Encoder ##
class EncoderArchitecture(nn.Module):
    def __init__(self, image_size, latent_size, hidden_size1=512, hidden_size2=256):
        super(EncoderArchitecture, self).__init__()
        self.l1 = nn.Linear(image_size, hidden_size2)
        self.l2 = nn.Linear(hidden_size2, hidden_size1)
        self.f1 = nn.ReLU()
        self.f2 = nn.ReLU()
        self.l3 = nn.Linear(hidden_size1, latent_size)  # Latent mean output
        self.l4 = nn.Linear(hidden_size1, latent_size)  # Latent log sd output
        self.softplus = nn.Softplus()

    def __call__(self, x):
        h0 = self.f1(self.l1(x[:, :, 0])) #TODO: to be fixed
        h1 = self.f2(self.l2(h0))
        output_mean = self.l3(h1)
        output_log_sd = self.l4(h1)
        return {"mean": output_mean, "sd": self.softplus(output_log_sd) + 0.1}


## Decoder ##
class DecoderArchitecture(nn.Module):
    def __init__(self, latent_size, image_size, hidden_size1=512, hidden_size2=256):
        super(DecoderArchitecture, self).__init__()
        self.l1 = nn.Linear(latent_size, hidden_size1)
        self.l2 = nn.Linear(hidden_size1, hidden_size2)
        self.f1 = nn.ReLU()
        self.f2 = nn.ReLU()
        self.l3 = nn.Linear(hidden_size2, image_size) # Latent mean output
        self.softplus = nn.Softplus()

    def __call__(self, x):
        h0 = self.f1(self.l1(x))
        h1 = self.f2(self.l2(h0))
        output_mean = self.l3(h1)
        return {"mean": output_mean}


# Initialize encoder and decoders
encoder = BF.BrancherFunction(EncoderArchitecture(image_size=image_size, latent_size=latent_size))
decoder = BF.BrancherFunction(DecoderArchitecture(latent_size=latent_size, image_size=image_size))

# Generative model
z = NormalVariable(np.zeros((latent_size,)), np.ones((latent_size,)), name="z")
decoder_output = DeterministicNode(decoder(z), name="decoder_output")
decoder_variance = DeterministicVariable(1., name="var")
x = BinomialVariable(n=1, logit_p=decoder_output["mean"], name="x")
model = ProbabilisticModel([x, z])

# Amortized variational distribution
Qx = EmpiricalVariable(dataset, batch_size=100, name="x", is_observed=True)
encoder_output = DeterministicNode(encoder(Qx), name="encoder_output")
Qz = NormalVariable(encoder_output["mean"], encoder_output["sd"], name="z")
model.set_posterior_model(ProbabilisticModel([Qx, Qz]))

# Joint-contrastive inference
inference.perform_inference(model,
                            number_iterations=3000,
                            number_samples=1,
                            optimizer="Adam",
                            lr=0.001)
loss_list = model.diagnostics["loss curve"]

#Plot results
plt.plot(loss_list)
plt.show()

image_grid = []
num_res = 100
z_range = np.linspace(-3, 3, 30)
for z1 in z_range:
    image_row = []
    for z2 in z_range:
        sample = model.get_sample(num_res, input_values={z: np.array([z1, z2]), decoder_variance: 0.000000001})
        image = np.reshape(sample["decoder_output"], newshape=(28, 28))
        image_row += [image]
    image_grid += [np.concatenate(image_row, axis=0)]
image_grid = np.concatenate(image_grid, axis=1)
plt.imshow(image_grid)
plt.colorbar()
plt.show()

#for _ in range(10):
#    sample = model.get_sample(1, input_values={decoder_variance: 0.000000001})
#    plt.imshow(np.reshape(sample["x"][0], newshape=(28, 28)))
#    plt.show()