"""
Links
---------
Module description
"""
import inspect

import chainer.functions as F
import chainer.links as L
from chainer import Link, Chain, ChainList

from brancher.functions import BrancherFunction


# class GaussianLinearRegressionLink(ChainList):
#     """
#     Summary
#
#     Parameters
#     ----------
#     num_regressors : int
#         Summary
#     """
#     def __init__(self, regressors, log_var, num_regressors):
#         self.regressors = regressors
#         self.log_var = log_var
#         links = [L.Linear(num_regressors, 1)]
#         super(GaussianLinearRegressionLink, self).__init__(*links)
#
#     def __call__(self, input_data):
#         """
#         Summary
#         """
#         try:
#             x, log_var = input_data[self.regressors], input_data[self.log_var]
#         except KeyError:
#             raise KeyError("The input parents do not match the expected parents")
#         mu = self[0](x)
#         return {"mean": mu, "var": F.exp(log_var)}
