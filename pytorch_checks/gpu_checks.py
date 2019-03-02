# import numpy as np
# from brancher.config import set_device
#
# # Set device
# set_device('cpu')
# from brancher.config import device
# print('Current device: ' + device.type)
#
#
# import numpy as np
#

#
# x = np.random.normal(size=(10, 3))
# T = coerce_to_dtype(x)
# T.device

##
import numpy as np

from brancher.variables import ProbabilisticModel
from brancher.standard_variables import NormalVariable, LaplaceVariable, CauchyVariable, LogNormalVariable
from brancher import inference
from brancher.config import device
from brancher.utilities import coerce_to_dtype
print('Current device: ' + device.type)

# # Real model
# nu_real = 1.
# mu_real = -2.
# x_real = LaplaceVariable(mu_real, nu_real, "x_real")
# data = x_real._get_sample(number_samples=50)


##
import pandas as pd
x = np.random.normal(size=(10, 3))
d = pd.DataFrame(x)
t = coerce_to_dtype(d)
print(type(t))