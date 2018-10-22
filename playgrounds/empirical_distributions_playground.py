import numpy as np

import chainer
import chainer.links as L
import chainer.functions as F

from brancher.variables import DeterministicVariable, RandomVariable, ProbabilisticModel
from brancher.standard_variables import NormalVariable, EmpiricalVariable
from brancher.functions import BrancherFunction
import brancher.functions as BF

## Data ##
dataset_size = 100
number_dimensions = 4
dataset = np.random.normal(0, 1, (dataset_size, number_dimensions))

## Variables ##
a = EmpiricalVariable(dataset, batch_size=5, is_observed=True, name='a')

## Sample ##
samples1 = a.get_sample(1)
samples2 = a.get_sample(1)

print(samples1[a])
print(samples2[a])