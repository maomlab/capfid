"""Per-example Jacobians of model outputs with respect to (a subset of) parameters."""
from __future__ import annotations

from typing import Callable, Optional

import torch
from torch.func import functional_call, jacrev, vmap

__all__ = ["selected_params", "flat_params", "jacobians"]

ParamFilter = Optional[Callable[[str], bool]]


def selected_params(model: torch.nn.Module, param_filter: ParamFilter = None) -> dict:
    """Trainable parameters whose name passes ``param_filter`` (all if None)."""
    return {n: p for n, p in model.named_parameters()
            if p.requires_grad and (param_filter is None or param_filter(n))}


def flat_params(model: torch.nn.Module, param_filter: ParamFilter = None) -> torch.Tensor:
    """Selected parameters concatenated into one vector, in ``named_parameters`` order."""
    return torch.cat([p.detach().reshape(-1) for p in selected_params(model, param_filter).values()])


def jacobians(model: torch.nn.Module, X: torch.Tensor, param_filter: ParamFilter = None,
              batch_size: int = 128) -> torch.Tensor:
    """Jacobians ``J[i] = ∂f(x_i)/∂θ``, shape (N, C, P), returned on CPU in the model dtype.

    The model is evaluated one example at a time (``vmap``), so it must map a batch of shape
    (B, ...) to outputs of shape (B, C) without mixing examples (no batch norm in train mode).
    """
    model.eval()
    params = {k: v.detach() for k, v in selected_params(model, param_filter).items()}
    if not params:
        raise ValueError("param_filter selected no parameters")
    frozen = {k: v.detach() for k, v in model.named_parameters() if k not in params}
    buffers = dict(model.named_buffers())

    def f(p, x):
        return functional_call(model, {**p, **frozen, **buffers}, (x.unsqueeze(0),)).squeeze(0)

    jf = vmap(jacrev(f), in_dims=(None, 0))
    out = []
    for i in range(0, len(X), batch_size):
        Jd = jf(params, X[i:i + batch_size])
        out.append(torch.cat([Jd[k].flatten(2) for k in params], -1).cpu())
    return torch.cat(out)
