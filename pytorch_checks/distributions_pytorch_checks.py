import torch
import numpy as np
from torch import distributions
from brancher import utilities
from importlib import reload

##
mu, sigma, x = torch.zeros(3, 1), torch.ones(3, 1), torch.randn(3, 1)
mu, sigma, x = utilities.broadcast_and_squeeze(mu, sigma, x)

print([i.numpy().shape for i in [mu, sigma, x]])

old = -0.5*torch.log(2*np.pi*sigma**2) - 0.5*(x-mu)**2/(sigma**2)
new = distributions.normal.Normal(loc=mu, scale=sigma).log_prob(x)

print(torch.equal(old, new))
print(torch.equal(utilities.sum_data_dimensions(old), utilities.sum_data_dimensions(new)))

##
mu, sigma, x = torch.zeros(3, 1), torch.ones(3, 1), torch.randn(3, 1)
mean, var = utilities.broadcast_and_squeeze(mu, sigma)
old = mean + var*torch.tensor(np.random.normal(0, 1, size=mean.shape)).type(torch.FloatTensor)
new = distributions.normal.Normal(loc=mean, scale=var).sample()

print(old, new)

##