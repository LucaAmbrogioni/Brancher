import chainer
import matplotlib.pyplot as plt
import numpy as np

from brancher.variables import DeterministicVariable, RandomVariable, ProbabilisticModel
from brancher.standard_variables import NormalVariable, LogNormalVariable, LogitNormalVariable, \
    BinomialVariable, ConcreteVariable
from brancher import inference
import brancher.functions as BF

# Parameters
num_timepoints = 50
T = 1.
time_range = np.linspace(0, T, num_timepoints)
dt = T/float(num_timepoints)
kernel = lambda tau, ts = 2.: np.exp(-tau/ts)*tau


# Probabilistic model #
w12 = LogNormalVariable(0., 1., "w12")
w21 = LogNormalVariable(0., 1., "w21")
base_rate = DeterministicVariable(5., "base rate")

rates1 = []
spikes1 = [DeterministicVariable(np.array([[0.], [1.]]), "spike1(0)")]
rates2 = []
spikes2 = [DeterministicVariable(np.array([[0.], [1.]]), "spike2(0)")]
for n, t in enumerate(time_range):
    rates1.append(base_rate + sum([w21*s[:1, :]*kernel(t - ts) for ts, s in zip(time_range[:n-1], spikes2)]))
    #prob = BF.concat((rates1[n]*dt, 1.-rates1[n]*dt), axis=1)
    #spikes1.append(ConcreteVariable(tau=0.2, p=prob, name="spike1({})".format(t)))
    spikes1.append(BinomialVariable(n=1, p=rates1[n]*dt, name="spike1({})".format(t)))

    rates2.append(base_rate + sum([w12*s[:1, :]*kernel(t - ts) for ts, s in zip(time_range[:n - 1], spikes1)]))
    # prob = BF.concat((rates1[n]*dt, 1.-rates1[n]*dt), axis=1)
    # spikes1.append(ConcreteVariable(tau=0.2, p=prob, name="spike1({})".format(t)))
    spikes2.append(BinomialVariable(n=1, p=rates2[n]*dt, name="spike1({})".format(t)))
model = ProbabilisticModel(spikes1 + spikes2)

# Samples
samples = model.get_sample(1)
pass