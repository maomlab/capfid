"""capfid — per-example model capacity versus fidelity for PyTorch models.

How many bits of a trained network's capacity does each training example (or set of examples)
consume, and how well is it fit? capfid answers this with a linearized Laplace posterior:
an example's capacity is how much the weight posterior changes when it is unlearned.

    >>> la = capfid.LinearizedLaplace(model, X, y, capfid.Categorical(), prior_precision=alpha)
    >>> res = la.pointwise()          # dict of per-example arrays
    >>> la.set_capacity(indices)      # joint capacity of a subset
    >>> la.totals()                   # effective params, model information, MDL code length
"""
from . import plotting
from .jacobians import flat_params, jacobians
from .laplace import LinearizedLaplace, Totals
from .likelihoods import Categorical, Gaussian, Likelihood
from .training import map_objective, prior_precision_from_weight_decay, retrain_without, train_map

__version__ = "0.1.0"

__all__ = [
    "LinearizedLaplace", "Totals", "Likelihood", "Categorical", "Gaussian",
    "jacobians", "flat_params", "map_objective", "train_map", "retrain_without",
    "prior_precision_from_weight_decay", "__version__",
]
