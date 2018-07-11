"""
Chains
---------
Module description
"""
from chainer import ChainList


class EmptyChain(ChainList):
    """
    Summary
    """
    def __init__(self):
        links = []
        super(EmptyChain, self).__init__(*links)
