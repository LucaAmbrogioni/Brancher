import torch
import chainer
import chainer.functions as F
import numpy as np
from brancher import utilities
from importlib import reload

def equal_tensor_variable(tensor, variable):
    return np.all(np.equal(tensor.numpy(), variable.data))

## torch.cat
tensor_tuple = (torch.randn(10, 5), torch.randn(40, 5))
torch.cat(tensor_tuple, dim=0).shape

## tensor.usqueeze() for expanding dims
x = np.random.normal(size=(10, 5))
dim = 2

xt = torch.tensor(x).unsqueeze(dim=dim)
xc = F.expand_dims(x, axis=dim)

equal_tensor_variable(xt, xc)

##
outt = utilities.uniform_shapes(tensor_tuple[0], tensor_tuple[1])
outc = utilities.uniform_shapes_chainer(chainer.Variable(tensor_tuple[0].numpy()), chainer.Variable(tensor_tuple[1].numpy()))
all([equal_tensor_variable(xt, xc) for xt, xc in zip(outt, outc)])

## torch.repeat for tiling
ns = 20
x = np.random.normal(size=(1, 20))
xt = utilities.tile_parameter(torch.tensor(x), number_samples=ns)
xc = utilities.tile_parameter_chainer(chainer.Variable(x), number_samples=ns)

equal_tensor_variable(xt, xc)

## get_diagonal: torch reshape
x = np.random.normal(size=(1, 20, 10, 20))
xt = utilities.get_diagonal(torch.tensor(x))
xc = utilities.get_diagonal_chainer(chainer.Variable(x))

equal_tensor_variable(xt, xc)

## sum_from_dim: torch sum
x = np.random.normal(size=(20, 5, 4, 2))
dim_index = 1
xt = utilities.sum_from_dim(torch.tensor(x), dim_index)
xc = utilities.sum_from_dim_chainer(chainer.Variable(x), dim_index)

equal_tensor_variable(xt, xc)

## partial_broadcast
xl = []
for i in range(1,3):
    xl.append(np.random.normal(size=(20, i, 10, 3)))

xt = utilities.partial_broadcast(*[torch.tensor(x) for x in xl])
xc = utilities.partial_broadcast_chainer(*[chainer.Variable(x) for x in xl])

print([i.shape for i in xl])
print([i.numpy().shape for i in xt])
print([i.shape for i in xc])

##

