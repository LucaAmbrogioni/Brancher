import chainer
import chainer.functions as F
import chainer.links as L
import matplotlib.pyplot as plt
import numpy as np

from brancher.variables import DeterministicVariable, ProbabilisticModel
from brancher.standard_variables import NormalVariable, EmpiricalVariable
from brancher import inference
import brancher.functions as BF

# Data
image_size = 28*28
latent_size = 20

train, test = chainer.datasets.get_mnist()
dataset_size = len(train)
dataset = np.array([np.reshape(image[0], newshape=(image_size, 1))
                    for image in train]).astype("float32") # TODO: Try without reshape when everything work

# Neural architectures


## Encoder ##
class EncoderArchitecture(chainer.ChainList):
    def __init__(self, image_size, latent_size, hidden_size=50):
        links = [L.Linear(in_size=image_size, out_size=hidden_size),
                 L.Linear(in_size=hidden_size, out_size=latent_size), # Latent mean output
                 L.Linear(in_size=hidden_size, out_size=latent_size)] # Latent log sd output
        super(EncoderArchitecture, self).__init__(*links)

    def __call__(self, x):
        h = F.relu(self[0](x))
        output_mean = self[1](h)
        output_log_sd = self[2](h)
        return {"mean": output_mean, "sd": F.softplus(output_log_sd)}


## Decoder ##
class DecoderArchitecture(chainer.ChainList):
    def __init__(self, latent_size, image_size, hidden_size=50):
        links = [L.Linear(in_size=latent_size, out_size=hidden_size),
                 L.Linear(in_size=hidden_size, out_size=image_size), # Pixel mean output
                 L.Linear(in_size=hidden_size, out_size=image_size)] # Pixel log sd output
        super(DecoderArchitecture, self).__init__(*links)

    def __call__(self, z):
        h = F.relu(self[0](z))
        output_mean = self[1](h)
        output_log_sd = self[2](h)
        return {"mean": output_mean, "sd": F.softplus(output_log_sd)}


# Initialize encoder and decoders
encoder = BF.BrancherFunction(EncoderArchitecture(image_size=image_size, latent_size=latent_size))
decoder = BF.BrancherFunction(DecoderArchitecture(latent_size=latent_size, image_size=image_size))

# Generative model
z = NormalVariable(np.zeros((latent_size,)), np.ones((latent_size,)), name="z")
decoder_output = decoder(z)
x = NormalVariable(decoder_output["mean"], decoder_output["sd"], name="x")
model = ProbabilisticModel([x, z])

# Amortized variational distribution
Qx = EmpiricalVariable(dataset, batch_size=50, name="x")
encoder_output = encoder(Qx)
Qz = NormalVariable(encoder_output["mean"], encoder_output["sd"], name="z")
model.set_posterior_model(ProbabilisticModel([Qx, Qz]))

# Joint-contrastive inference
inference.stochastic_variational_inference(model,
                                           number_iterations=2000,
                                           number_samples=1,
                                           optimizer=chainer.optimizers.Adam(0.001))
loss_list = model.diagnostics["loss curve"]

#Plot results
plt.plot(loss_list)
plt.show()

sample = model.get_sample(1)
plt.imshow(np.reshape(sample["x"][0], newshape=(28, 28)))
plt.show()

#post_sample = model.get_posterior_sample(1)
#print(post_sample["z"][0])
#plt.imshow(np.reshape(post_sample["x"][0][0,:], newshape=(28, 28)))
#plt.show()