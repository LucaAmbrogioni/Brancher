import chainer
import matplotlib.pyplot as plt
import numpy as np

from brancher.variables import DeterministicVariable, ProbabilisticModel
from brancher.standard_variables import NormalVariable, EmpiricalVariable
from brancher import inference
import brancher.functions as BF

# Data

# Neural architectures
#Encoder
#Decoder

# Generative model
latent_size = (10,)
z = NormalVariable(np.zeros(latent_size), np.ones(latent_size))
decoder_output = decoder(z)
x = NormalVariable(decoder_output["mean"], BF.exp(decoder_output["log_var"]), name="x")
model = ProbabilisticModel([x, z])

# Amortized variational distribution
Qx = EmpiricalVariable(dataset, name="x")
encoder_output = encoder(Qx)
Qz = NormalVariable(decoder_output["mean"], BF.exp(decoder_output["log_var"]), name="z")
variational_model = ProbabilisticModel([Qx, Qz])

