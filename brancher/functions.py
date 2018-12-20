import types

import chainer
import chainer.functions as F

from brancher.variables import var2link
from brancher.variables import Variable, PartialLink


class BrancherFunction(object):

    def __init__(self, fn):
        self.fn = fn
        if isinstance(fn, (chainer.Link, chainer.Chain, chainer.ChainList)):
            self.links = {fn}
        else:
            self.links = set()

    def __call__(self, *args, **kwargs):
        link_args = [var2link(arg) for arg in args]
        link_kwargs = {name: var2link(arg) for name, arg in kwargs.items()}
        arg_vars = {var for link in link_args if isinstance(link, PartialLink) for var in link.vars}
        kwarg_vars = {var for _, link in link_kwargs.items() if isinstance(link, PartialLink) for var in link.vars}

        def fn(values):
            args = [x.fn(values) if isinstance(x, PartialLink) else x for x in link_args]
            kwargs = dict({(name, x.fn(values)) if isinstance(x, PartialLink) else (name, x)
                           for name, x in link_kwargs.items()})
            return self.fn(*args, **kwargs)

        return PartialLink(arg_vars.union(kwarg_vars), fn, self.links)

    @staticmethod
    def _is_var(self, arg):
        return isinstance(arg, (Variable, PartialLink))


is_chainer_fn = lambda k, v: type(v) is types.FunctionType and not k.startswith('_') #TODO: Work in progress
brancher_fns = {name: BrancherFunction(v) for name, v in F.__dict__.items() if is_chainer_fn(name, v)}
globals().update(brancher_fns)
