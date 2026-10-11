r"""Empirical neural tangent kernel (NTK) Gram matrices, G = J Jᵀ.

``KernelLaplace`` needs only inner products of per-example Jacobians,

    G[(i, c), (j, d)] = ∂f_c(x_i)/∂θ · ∂f_d(x'_j)/∂θ,

with rows ordered example-major (i·C + c), matching ``J.reshape(N * C, P)``.

For a linear layer z = W a + b, the Jacobian of output c with respect to W is the outer product
δ_c aᵀ of the backpropagated output gradient δ_c = ∂f_c/∂z and the layer input a. Hence

    J(x)·J(x')ᵀ restricted to (W, b) = (δ·δ')(a·a' + 1),

and when every selected parameter belongs to an ``nn.Linear`` layer applied to (B, d) inputs
(an MLP, or a last-layer posterior) the Gram matrix is assembled exactly from layer inputs and
output gradients, without forming P-dimensional Jacobians. Otherwise per-example Jacobians are
used (memory N·C·P).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .jacobians import ParamFilter, jacobians, selected_params

__all__ = ["ntk_gram"]

DT = torch.float64


def _linear_modules(model, param_filter):
    """{module: (has_weight, has_bias)} if every selected parameter belongs to an nn.Linear."""
    sel = selected_params(model, param_filter)
    if not sel:
        raise ValueError("param_filter selected no parameters")
    mods, found = {}, 0
    for mname, m in model.named_modules():
        for pname, _ in m.named_parameters(recurse=False):
            if (f"{mname}.{pname}" if mname else pname) not in sel:
                continue
            if not isinstance(m, nn.Linear):
                return None
            w, b = mods.get(m, (False, False))
            mods[m] = (w or pname == "weight", b or pname == "bias")
            found += 1
    return mods if found == len(sel) else None


def _linear_factors(model, X, mods, batch_size):
    """Per selected Linear layer: inputs a (N, d_in) and output gradients δ (N, C, d_out)."""
    acts = {m: [] for m in mods}
    grads = {m: [] for m in mods}
    for i in range(0, len(X), batch_size):
        store, calls = {}, {m: 0 for m in mods}

        def hook(mod, inp, out):
            calls[mod] += 1
            store[mod] = (inp[0], out)

        handles = [m.register_forward_hook(hook) for m in mods]
        try:
            with torch.enable_grad():
                f = model(X[i:i + batch_size])
        finally:
            for h in handles:
                h.remove()
        if any(c != 1 for c in calls.values()) or any(store[m][0].dim() != 2 for m in mods):
            return None                                     # shared layers or sequence inputs
        outs = [store[m][1] for m in mods]
        per_c = [torch.autograd.grad(f[:, c].sum(), outs, retain_graph=True)
                 for c in range(f.shape[1])]
        for k, m in enumerate(mods):
            acts[m].append(store[m][0].detach().to(DT).cpu())
            grads[m].append(torch.stack([g[k] for g in per_c], 1).detach().to(DT).cpu())
    return {m: (torch.cat(acts[m]), torch.cat(grads[m])) for m in mods}


def _gram_from_factors(mods, fa, fb):
    G = None
    for m, (has_w, has_b) in mods.items():
        a1, d1 = fa[m]
        a2, d2 = fb[m]
        N1, C, _ = d1.shape
        N2 = d2.shape[0]
        dd = (d1.reshape(N1 * C, -1) @ d2.reshape(N2 * C, -1).T)              # δ·δ'
        aa = (a1 @ a2.T if has_w else torch.zeros(N1, N2, dtype=DT)) + float(has_b)
        term = dd * aa.repeat_interleave(C, 0).repeat_interleave(C, 1)
        G = term if G is None else G + term
    return G


def ntk_gram(model: nn.Module, X: torch.Tensor, X2: torch.Tensor = None,
             param_filter: ParamFilter = None, batch_size: int = 512,
             method: str = "auto") -> torch.Tensor:
    """Empirical NTK Gram matrix ``J(X) J(X2)ᵀ``, shape (N·C, N2·C), float64 on CPU.

    Parameters
    ----------
    model : torch.nn.Module
        Maps (B, ...) inputs to (B, C) outputs without mixing examples; put in eval mode.
    X, X2 : torch.Tensor
        Inputs; ``X2`` defaults to ``X`` (the training Gram matrix).
    param_filter : callable(name) -> bool, optional
        Restrict to these parameters, as in ``LinearizedLaplace``.
    method : {"auto", "linear", "jacobian"}
        "linear" uses the exact factorization for ``nn.Linear`` layers (see module docstring);
        "jacobian" forms per-example Jacobians; "auto" uses "linear" when it applies.
    """
    model.eval()
    same = X2 is None
    if method not in ("auto", "linear", "jacobian"):
        raise ValueError(f"unknown method {method!r}")
    if method != "jacobian":
        mods = _linear_modules(model, param_filter)
        fa = None if mods is None else _linear_factors(model, X, mods, batch_size)
        fb = fa if same or fa is None else _linear_factors(model, X2, mods, batch_size)
        if fa is not None and fb is not None:
            return _gram_from_factors(mods, fa, fb)
        if method == "linear":
            raise ValueError("method='linear' needs every selected parameter in an nn.Linear "
                             "layer applied once to (B, d) inputs")
    Ja = jacobians(model, X, param_filter, batch_size).to(DT)
    Ja = Ja.reshape(-1, Ja.shape[-1])
    if same:
        return Ja @ Ja.T
    Jb = jacobians(model, X2, param_filter, batch_size).to(DT)
    return Ja @ Jb.reshape(-1, Jb.shape[-1]).T
