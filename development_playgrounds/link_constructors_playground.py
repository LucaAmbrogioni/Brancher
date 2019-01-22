
#a = DeterministicVariable(0.5, 'a')
#b = DeterministicVariable(0.3, 'b')
#c = DeterministicVariable(0.3, 'b')
#d = NormalVariable(F.sin(a*b + c), c + a, 'd')

# Expected result
#RandomVariable(distribution=NormalDistribution(),
#               name="d",
#               parents=(a, b, c),
#               link=lambda values: {'mu': F.sin(values[a]*values[b] + values[c]), 'sigma': values[c] + values[a]})
import numpy as np

#import chainer.functions as F
import chainer
import chainer.links as L
import chainer.functions as F

#from brancher.links import brancher_decorator
from brancher.variables import DeterministicVariable, RandomVariable, ProbabilisticModel
from brancher.standard_variables import NormalVariable
from brancher.functions import BrancherFunction
import brancher.functions as BF
#import brancher.links as BL

a = DeterministicVariable(data=1.5, name='a', learnable=True)
##
b = DeterministicVariable(0.3, 'b')
c = DeterministicVariable(0.3, 'c')
d = NormalVariable((a*b + c), c + a**2, 'd')
e1 = BF.concat((a, b), 2)
e2 = BF.concat((a, c), 2)
f = NormalVariable(e1**2, e2**1, 'f')

f._get_sample(10)

a_val = chainer.Variable(0.25*np.pi*np.ones((1,1), dtype = "float32"))
b_val = chainer.Variable(0.25*np.pi*np.ones((1,1), dtype = "float32"))
c_val = chainer.Variable(2*np.ones((1,1), dtype = "float32"))

#z = BF.sin(a + b)/c

#print(z.fn({a: a_val, b: b_val, c: c_val}))

BLink = BrancherFunction(L.Linear(1, 10))

print(BLink)
#import inspect
#print(inspect.getmro(BLink))
#print(issubclass(BLink, chainer.Link))

print(BLink(a).fn({a: a_val}).data)

pass