import torch
import chainer
import chainer.functions as F
import numpy as np
from brancher import utilities
from importlib import reload

is_observed = False

##
def check_value_datatype(t):
    print(t)
    print(type(t))
    print('Shape: ', t.shape)
    print('Dtype: ', t.dtype)


##
data = 1
t = utilities.coerce_to_dtype(data, is_observed)
check_value_datatype(t)

##
data = 1.4
utilities.coerce_to_dtype(data, is_observed)


##
data = np.float64(4.3)
utilities.coerce_to_dtype(data, is_observed)


##
data = np.int8(4.3)
utilities.coerce_to_dtype(data, is_observed)


##
data = np.random.normal(size=(10, 12, 12))
utilities.coerce_to_dtype(data, is_observed)

##
data = torch.tensor(np.random.normal(size=(10, 12, 12)))
utilities.coerce_to_dtype(data, is_observed)