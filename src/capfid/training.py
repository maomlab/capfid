"""MAP training and leave-out retraining matching the posterior's objective.

capfid only needs a model trained to (near) a stationary point of

    L(θ) = Σ_i ℓ(y_i, f(x_i; θ)) + (α/2)‖θ‖².

Train it any way you like; these full-batch helpers are for small models and for validating
the one-step (Newton) unlearning against actual retraining.
"""
from __future__ import annotations

import copy

import torch

__all__ = ["map_objective", "train_map", "retrain_without", "prior_precision_from_weight_decay"]


def prior_precision_from_weight_decay(weight_decay: float, n_train: int) -> float:
    """α for a model trained on the *mean* loss with coupled L2 weight decay λ.

    SGD/Adam with ``weight_decay=λ`` on a mean loss follows the gradient of
    (1/N)[Σℓ + (Nλ/2)‖θ‖²], so α = N·λ. AdamW's decoupled decay is only approximately L2.
    """
    return float(n_train) * float(weight_decay)


def map_objective(model, X, y, likelihood, prior_precision, keep=None):
    """Summed negative log-likelihood (optionally over ``keep``) plus (α/2)‖θ‖²."""
    nll = likelihood.nll(model(X), y)
    if keep is not None:
        nll = nll[keep]
    reg = sum((p ** 2).sum() for p in model.parameters() if p.requires_grad)
    return nll.sum() + 0.5 * prior_precision * reg


def train_map(model, X, y, likelihood, prior_precision, adam_steps=2000, lr=3e-3,
              lbfgs_iters=300, keep=None):
    """Full-batch Adam (cosine schedule) then L-BFGS polishing to a near-stationary point."""
    model.train()
    if adam_steps:
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, adam_steps)
        for _ in range(adam_steps):
            opt.zero_grad()
            map_objective(model, X, y, likelihood, prior_precision, keep).backward()
            opt.step()
            sched.step()
    if lbfgs_iters:
        opt = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=lbfgs_iters,
                                tolerance_grad=1e-9, tolerance_change=1e-12,
                                history_size=50, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            loss = map_objective(model, X, y, likelihood, prior_precision, keep)
            loss.backward()
            return loss

        opt.step(closure)
    model.eval()
    return model


def retrain_without(model, X, y, likelihood, prior_precision, idx, lbfgs_iters=300):
    """Copy of ``model`` retrained (warm start, L-BFGS) with examples ``idx`` removed."""
    keep = torch.ones(len(X), dtype=torch.bool)
    keep[torch.as_tensor(idx)] = False
    return train_map(copy.deepcopy(model), X, y, likelihood, prior_precision,
                     adam_steps=0, lbfgs_iters=lbfgs_iters, keep=keep)
