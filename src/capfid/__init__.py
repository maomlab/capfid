"""capfid — per-example model capacity versus fidelity for PyTorch models.

How many bits of a trained network's capacity does each training example (or set of examples)
consume, and how well is it fit? capfid answers this with a linearized Laplace posterior:
an example's capacity is how much the weight posterior changes when it is unlearned.

    >>> la = capfid.LinearizedLaplace(model, X, y, capfid.Categorical(), prior_precision=alpha)
    >>> res = la.pointwise()          # dict of per-example arrays
    >>> la.set_capacity(indices)      # joint capacity of a subset
    >>> la.totals()                   # effective params, model information, MDL code length

For wide networks (parameters P much larger than N·C), ``capfid.KernelLaplace`` computes the
same posterior in output space through the empirical neural tangent kernel.
"""
from . import plotting
from .jacobians import flat_params, jacobians
from .laplace import KernelLaplace, LinearizedLaplace, Totals
from .likelihoods import Bernoulli, Categorical, Gaussian, Likelihood
from .ntk import ntk_gram
from .training import map_objective, prior_precision_from_weight_decay, retrain_without, train_map

__version__ = "1.0.0"

__all__ = [
    "LinearizedLaplace", "KernelLaplace", "Totals", "Likelihood", "Categorical", "Bernoulli",
    "Gaussian", "jacobians", "flat_params", "ntk_gram", "map_objective", "train_map",
    "retrain_without", "prior_precision_from_weight_decay", "__version__",
]
