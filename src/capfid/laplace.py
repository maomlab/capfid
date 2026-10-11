r"""Linearized Laplace posterior with per-example and per-set capacity and fidelity.

Model and posterior
-------------------
The network is assumed trained (to a MAP estimate θ*) on the *summed* objective

    L(θ) = Σ_i ℓ(y_i, f(x_i; θ)) + (α/2)‖θ‖²,       α = prior precision.

The linearized Laplace (generalized Gauss–Newton) posterior is q_D = N(θ*, Σ) with

    Σ⁻¹ = A = Σ_i J_iᵀ Λ_i J_i + α I,

J_i = ∂f(x_i)/∂θ, Λ_i = ∂²ℓ/∂f², r_i = ∂ℓ/∂f. Removing a set S ("unlearning") subtracts its
curvature, A_{-S} = A − J_Sᵀ Λ_S J_S, and moves the mean by one Newton step,
θ*_{-S} = θ* + A_{-S}⁻¹ J_Sᵀ r_S.

Capacity of S (all reduce to mC × mC matrices; see the manuscript, Proposition 1)
----------------------------------------------------------------------------------
    dof          tr K_S                       effective number of parameters spent on S
    volume       h(q_{D\S}) − h(q_D)          = −½ log det(I − K_S); label-agnostic
    information  KL(q_D ‖ q_{D\S})            = ½[r_Sᵀ S_{-S} r_S − tr K_S − log det(I − K_S)]

with S_S = J_S Σ J_Sᵀ, K_S = Λ^{½} S_S Λ^{½}, S_{-S} = S_S + S_S Λ^{½}(I − K_S)⁻¹ Λ^{½} S_S.

Fidelity
--------
    fidelity          log p(y | x, θ*)
    heldout_fidelity  log p(y | x, θ*_{-i}), using linearized held-out outputs f + S_{-i} r_i

Under exact Bayesian updating, information = E_{q_D} log p(y) − log E_{q_{D\S}} p(y)
(the generalization gap); this is exact for linear-Gaussian models.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import torch

from .jacobians import ParamFilter, flat_params, jacobians
from .likelihoods import Likelihood, psd_sqrt

__all__ = ["LinearizedLaplace", "Totals"]

LN2 = math.log(2.0)
DT = torch.float64


@dataclass
class Totals:
    """Whole-model quantities. The Laplace evidence is read as a two-part code:
    −log2 p(D) ≈ model_bits + data_bits."""

    n_params: int
    dof: float                # γ = tr(Σ GGN), MacKay's effective number of parameters
    volume_bits: float        # h(prior) − h(q_D) = ½ log det(I + GGN/α)
    model_bits: float         # α/2 ‖θ*‖² + volume  (model code length)
    data_bits: float          # −Σ log2 p(y_i | x_i, θ*)  (data code length)
    kl_prior_bits: float      # KL(q_D ‖ prior) = model − γ/2 nats
    log_evidence_nats: float  # Laplace approximation to log p(D)
    unique_fraction: float    # Σ_i volume({i}) / volume_total  (≤ 1 by submodularity)

    @property
    def description_length_bits(self) -> float:
        return self.model_bits + self.data_bits

    def as_dict(self) -> dict:
        return {**asdict(self), "description_length_bits": self.description_length_bits}


def _removal_terms(S, Lh, r):
    """Batched removal algebra. S, Lh: (n, k, k); r: (n, k).
    Returns tr K, log det(I − K), S_{-S}, r ᵀ S_{-S} r (each batched)."""
    K = Lh @ S @ Lh
    IK = torch.eye(S.shape[-1], dtype=S.dtype) - K
    logdet_IK = torch.linalg.slogdet(IK)[1]
    S_loo = S + S @ Lh @ torch.linalg.solve(IK, Lh @ S)
    S_loo = 0.5 * (S_loo + S_loo.transpose(-1, -2))
    trK = K.diagonal(dim1=-2, dim2=-1).sum(-1)
    quad = (r.unsqueeze(-2) @ S_loo @ r.unsqueeze(-1))[..., 0, 0]
    return trK, logdet_IK, S_loo, quad


class LinearizedLaplace:
    """Linearized Laplace posterior around a trained model, with capacity/fidelity scores.

    Parameters
    ----------
    model : torch.nn.Module
        Trained network mapping a batch of inputs (B, ...) to outputs (B, C). It is put in
        eval mode; it must not mix examples within a batch.
    X, y : torch.Tensor
        Training inputs and targets (the data the model was trained on).
    likelihood : capfid.Likelihood
        e.g. ``Categorical()`` or ``Gaussian(sigma)``.
    prior_precision : float
        α, the L2 coefficient of the *summed* training objective (see
        ``prior_precision_from_weight_decay`` for mean-loss training).
    param_filter : callable(name) -> bool, optional
        Restrict the posterior to these parameters (e.g. ``lambda n: n.startswith("head")``
        for a last-layer posterior); other parameters are held fixed.
    batch_size : int
        Batch size for forward passes and Jacobians.
    max_params : int
        Safety limit; the exact posterior needs O(P²) memory and O(P³) time.
    """

    def __init__(self, model, X, y, likelihood: Likelihood, prior_precision: float, *,
                 param_filter: ParamFilter = None, batch_size: int = 128, max_params: int = 20_000):
        if prior_precision <= 0:
            raise ValueError("prior_precision must be > 0")
        P = flat_params(model, param_filter).numel()
        if P > max_params:
            raise ValueError(
                f"{P} parameters selected; the exact posterior needs ~{8 * P * P / 1e9:.1f} GB. "
                "Use param_filter (e.g. last layer only) or raise max_params.")
        self.model, self.likelihood, self.alpha = model, likelihood, float(prior_precision)
        model.eval()
        with torch.no_grad():
            f = torch.cat([model(X[i:i + batch_size]) for i in range(0, len(X), batch_size)])
        self.f = f.detach().cpu().to(DT)
        y = y.detach().cpu()
        self.y = y.to(DT) if y.is_floating_point() else y
        self.r, self.Lam = likelihood.grad_hess(self.f, self.y)
        self._Lh = psd_sqrt(self.Lam)

        self.J = jacobians(model, X, param_filter, batch_size)          # (N, C, P), CPU
        self.theta = flat_params(model, param_filter).detach().cpu().to(DT)
        N, C, P = self.J.shape
        self.n, self.n_outputs, self.n_params = N, C, P

        A = self.alpha * torch.eye(P, dtype=DT)
        for i in range(0, N, batch_size):
            Jb = self.J[i:i + batch_size].to(DT)
            A += Jb.reshape(-1, P).T @ (self.Lam[i:i + batch_size] @ Jb).reshape(-1, P)
        self.L = torch.linalg.cholesky(A)
        self.logdet_A = 2.0 * self.L.diagonal().log().sum().item()
        del A

        S = torch.empty(N, C, C, dtype=DT)
        for i in range(0, N, batch_size):
            V = self._half_solve(self.J[i:i + batch_size].to(DT).reshape(-1, P).T).reshape(P, -1, C)
            S[i:i + batch_size] = torch.einsum("pbc,pbd->bcd", V, V)
        self.S = S                                                        # J_i Σ J_iᵀ

    def __repr__(self):
        return (f"LinearizedLaplace(n={self.n}, outputs={self.n_outputs}, params={self.n_params}, "
                f"prior_precision={self.alpha:g})")

    def _half_solve(self, B):
        return torch.linalg.solve_triangular(self.L, B, upper=False)

    @property
    def posterior_output_cov(self) -> torch.Tensor:
        """Epistemic covariance of the outputs at each training input, J_i Σ J_iᵀ, (N, C, C)."""
        return self.S

    # ------------------------------------------------------------------ single examples
    def pointwise(self, n_samples: int = 1000, seed: int = 0, chunk_size: int = 512,
                  return_logits: bool = False) -> dict:
        """Capacity and fidelity for every training example (removing one at a time).

        Returns a dict of 1-D numpy arrays (pass to ``pandas.DataFrame`` directly):

        ========================== ==========================================================
        dof                        effective parameters determined by the example (Σ = γ)
        volume_bits                h(θ | D∖i) − h(θ | D), label-agnostic capacity
        information_bits           KL(q_D ‖ q_{D∖i}), information stored about (x_i, y_i)
        shift_bits                 mean-shift (influence) part of information_bits
        fidelity_bits              log2 p(y_i | x_i, θ*)
        heldout_fidelity_bits      log2 p(y_i | x_i, θ*_{-i}) (one Newton step)
        memorization_bits          fidelity_bits − heldout_fidelity_bits
        expected_fidelity_bits     E_{q_D} log2 p(y_i | θ)
        predictive_fidelity_bits   log2 E_{q_D} p(y_i | θ)
        heldout_predictive_bits    log2 E_{q_{D∖i}} p(y_i | θ)
        ========================== ==========================================================

        The last three use ``n_samples`` Monte Carlo draws (closed form for ``Gaussian``).
        With ``return_logits=True`` the dict also holds ``heldout_outputs`` (N, C).
        """
        lik, N, C = self.likelihood, self.n, self.n_outputs
        trK, logdet_IK, S_loo, quad = _removal_terms(self.S, self._Lh, self.r)
        m_loo = self.f + (S_loo @ self.r.unsqueeze(-1)).squeeze(-1)

        g = torch.Generator().manual_seed(seed)
        e_ll, pred, ho_pred = [], [], []
        for i in range(0, N, chunk_size):
            sl = slice(i, min(i + chunk_size, N))
            eps = torch.randn(n_samples, sl.stop - sl.start, C, generator=g, dtype=DT)
            a, b = lik.gauss_expect(self.f[sl], self.S[sl], self.y[sl], eps)
            _, c = lik.gauss_expect(m_loo[sl], S_loo[sl], self.y[sl], eps)  # common random numbers
            e_ll.append(a), pred.append(b), ho_pred.append(c)

        fid = lik.log_lik(self.f, self.y)
        ho = lik.log_lik(m_loo, self.y)
        nb = lambda t: (t / LN2).numpy()  # noqa: E731
        out = dict(
            dof=trK.numpy(),
            volume_bits=nb(-0.5 * logdet_IK),
            information_bits=nb(0.5 * (quad - trK - logdet_IK)),
            shift_bits=nb(0.5 * quad),
            fidelity_bits=nb(fid),
            heldout_fidelity_bits=nb(ho),
            memorization_bits=nb(fid - ho),
            expected_fidelity_bits=nb(torch.cat(e_ll)),
            predictive_fidelity_bits=nb(torch.cat(pred)),
            heldout_predictive_bits=nb(torch.cat(ho_pred)),
        )
        if return_logits:
            out["heldout_outputs"] = m_loo.numpy()
        return out

    # ------------------------------------------------------------------ sets
    def set_capacity(self, idx, n_samples: int = 2000, seed: int = 0) -> dict:
        """Capacity and fidelity for removing the set ``idx`` jointly.

        Includes ``redundancy`` = volume / Σ single-example volumes (≥ 1; large when the
        members encode shared information). The joint information and held-out values rely on
        one Newton step for the whole set and become unreliable for large or heavily memorized
        sets; the volume depends only on curvature.
        """
        idx = torch.unique(torch.as_tensor(idx, dtype=torch.long).reshape(-1))
        m, C, P, lik = len(idx), self.n_outputs, self.n_params, self.likelihood
        V = self._half_solve(self.J[idx].to(DT).reshape(-1, P).T)
        S = V.T @ V
        Lh = torch.block_diag(*self._Lh[idx])
        r = self.r[idx].reshape(-1)
        trK, logdet_IK, S_loo, quad = (t[0] for t in _removal_terms(S[None], Lh[None], r[None]))
        f = self.f[idx]
        m_loo = f + (S_loo @ r).reshape(m, C)
        y = self.y[idx]

        g = torch.Generator().manual_seed(seed)
        eps = torch.randn(n_samples, m * C, generator=g, dtype=DT)
        e_ll, _ = lik.joint_expect(f, S, y, eps)
        _, ho_pred = lik.joint_expect(m_loo, S_loo, y, eps)

        t1, ld1, _, q1 = _removal_terms(self.S[idx], self._Lh[idx], self.r[idx])
        vol = -0.5 * logdet_IK
        vol_single = (-0.5 * ld1).sum()
        return dict(
            n=m,
            dof=trK.item(),
            volume_bits=(vol / LN2).item(),
            information_bits=(0.5 * (quad - trK - logdet_IK) / LN2).item(),
            sum_single_volume_bits=(vol_single / LN2).item(),
            sum_single_information_bits=(0.5 * (q1 - t1 - ld1).sum() / LN2).item(),
            redundancy=(vol / vol_single).item(),
            fidelity_bits=(lik.log_lik(f, y).sum() / LN2).item(),
            heldout_fidelity_bits=(lik.log_lik(m_loo, y).sum() / LN2).item(),
            expected_fidelity_bits=(e_ll / LN2).item(),
            heldout_predictive_bits=(ho_pred / LN2).item(),
        )

    # ------------------------------------------------------------------ whole model
    def totals(self) -> Totals:
        """Effective parameters, model information and the Laplace two-part code length.

        With ``param_filter``, the evidence is conditional on the frozen parameters."""
        trK, logdet_IK, _, _ = _removal_terms(self.S, self._Lh, self.r)
        gamma = trK.sum().item()
        P, a = self.n_params, self.alpha
        logdet_ratio = self.logdet_A - P * math.log(a)                  # log det(I + GGN/α)
        vol_tot = 0.5 * logdet_ratio
        model = 0.5 * a * float(self.theta @ self.theta) + vol_tot
        data = -float(self.likelihood.log_lik(self.f, self.y).sum())
        unique = float((-0.5 * logdet_IK).sum()) / vol_tot
        return Totals(P, gamma, vol_tot / LN2, model / LN2, data / LN2,
                      (model - 0.5 * gamma) / LN2, -(model + data), unique)
