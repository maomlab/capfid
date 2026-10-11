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

Two ways to compute the same posterior
--------------------------------------
``LinearizedLaplace`` factorizes the P × P matrix A (cost O(P³)). ``KernelLaplace`` works in
the N·C-dimensional output space with the empirical NTK G = J Jᵀ and the Woodbury identity,

    J Σ Jᵀ = α⁻¹ [G − G Λ^{½} M⁻¹ Λ^{½} G],    M = αI + Λ^{½} G Λ^{½},
    log det(I + GGN/α) = log det(M/α),

(cost O((NC)³)), which is the cheaper route for wide networks with P ≫ N·C.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch

from .jacobians import ParamFilter, flat_params, jacobians
from .likelihoods import Likelihood, psd_sqrt
from .ntk import ntk_gram

__all__ = ["LinearizedLaplace", "KernelLaplace", "Totals"]

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


class _Laplace:
    """Per-example, per-set and whole-model algebra shared by both posterior computations.

    Subclasses set ``self.S`` (N, C, C) = J_i Σ J_iᵀ and implement ``_set_cov(idx)`` (J_S Σ J_Sᵀ),
    ``_logdet_ratio()`` (log det(I + GGN/α)) and ``_new_moments(X)`` (posterior mean outputs
    and J(X) Σ J(X)ᵀ blocks at new inputs).
    """

    def _setup(self, model, X, y, likelihood, prior_precision, param_filter, batch_size):
        if prior_precision <= 0:
            raise ValueError("prior_precision must be > 0")
        self.model, self.likelihood, self.alpha = model, likelihood, float(prior_precision)
        self.param_filter, self.batch_size = param_filter, batch_size
        model.eval()
        self.f = self._forward(X)
        y = y.detach().cpu()
        self.y = y.to(DT) if y.is_floating_point() else y
        self.r, self.Lam = likelihood.grad_hess(self.f, self.y)
        self._Lh = psd_sqrt(self.Lam)
        self.theta = flat_params(model, param_filter).detach().cpu().to(DT)
        self.n, self.n_outputs = self.f.shape
        self.n_params = self.theta.numel()

    def _forward(self, X):
        with torch.no_grad():
            f = torch.cat([self.model(X[i:i + self.batch_size])
                           for i in range(0, len(X), self.batch_size)])
        return f.detach().cpu().to(DT)

    def __repr__(self):
        return (f"{type(self).__name__}(n={self.n}, outputs={self.n_outputs}, "
                f"params={self.n_params}, prior_precision={self.alpha:g})")

    @property
    def posterior_output_cov(self) -> torch.Tensor:
        """Epistemic covariance of the outputs at each training input, J_i Σ J_iᵀ, (N, C, C)."""
        return self.S

    def _prior_sq(self):
        """‖θ − θ₀‖² at the posterior mean (θ₀ = 0 unless the mean is refit)."""
        return float(self.theta @ self.theta)

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
        m, C, lik = len(idx), self.n_outputs, self.likelihood
        S = self._set_cov(idx)
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
        vol_tot = 0.5 * self._logdet_ratio()                             # ½ log det(I + GGN/α)
        model = 0.5 * a * self._prior_sq() + vol_tot
        data = -float(self.likelihood.log_lik(self.f, self.y).sum())
        unique = float((-0.5 * logdet_IK).sum()) / vol_tot
        return Totals(P, gamma, vol_tot / LN2, model / LN2, data / LN2,
                      (model - 0.5 * gamma) / LN2, -(model + data), unique)

    # ------------------------------------------------------------------ new inputs
    def predict(self, X_new, y_new=None, n_samples: int = 1000, seed: int = 0) -> dict:
        """Linearized posterior predictions at new inputs (e.g. a test set).

        Returns a dict of numpy arrays:

        ========================== ==========================================================
        outputs                    f(x; θ*), (N', C)
        output_cov                 J(x) Σ J(x)ᵀ, (N', C, C)
        volume_bits                h(q_D) − h(q_{D∪{x}}) = ½ log2 det(I + Λ^{½} J Σ Jᵀ Λ^{½}):
                                   the volume x would occupy if added to the training data
                                   with any label (Λ at f(x; θ*)); for a Gaussian likelihood,
                                   the mutual information between its label and the weights
        fidelity_bits              log2 p(y | x, θ*)            (only with ``y_new``)
        predictive_fidelity_bits   log2 E_{q_D} p(y | x, θ)     (only with ``y_new``)
        ========================== ==========================================================
        """
        lik = self.likelihood
        f, S = self._new_moments(X_new)
        Lh = psd_sqrt(lik.hess(f))
        I = torch.eye(f.shape[1], dtype=DT)
        vol = 0.5 * torch.linalg.slogdet(I + Lh @ S @ Lh)[1]
        out = dict(outputs=f.numpy(), output_cov=S.numpy(), volume_bits=(vol / LN2).numpy())
        if y_new is not None:
            y = y_new.detach().cpu()
            y = y.to(DT) if y.is_floating_point() else y
            g = torch.Generator().manual_seed(seed)
            eps = torch.randn(n_samples, *f.shape, generator=g, dtype=DT)
            _, pred = lik.gauss_expect(f, S, y, eps)
            out["fidelity_bits"] = (lik.log_lik(f, y) / LN2).numpy()
            out["predictive_fidelity_bits"] = (pred / LN2).numpy()
        return out


class LinearizedLaplace(_Laplace):
    """Linearized Laplace posterior around a trained model, with capacity/fidelity scores.

    Parameters
    ----------
    model : torch.nn.Module
        Trained network mapping a batch of inputs (B, ...) to outputs (B, C). It is put in
        eval mode; it must not mix examples within a batch.
    X, y : torch.Tensor
        Training inputs and targets (the data the model was trained on).
    likelihood : capfid.Likelihood
        e.g. ``Categorical()``, ``Bernoulli()`` or ``Gaussian(sigma)``.
    prior_precision : float
        α, the L2 coefficient of the *summed* training objective (see
        ``prior_precision_from_weight_decay`` for mean-loss training).
    param_filter : callable(name) -> bool, optional
        Restrict the posterior to these parameters (e.g. ``lambda n: n.startswith("head")``
        for a last-layer posterior); other parameters are held fixed.
    batch_size : int
        Batch size for forward passes and Jacobians.
    max_params : int
        Safety limit; the exact posterior needs O(P²) memory and O(P³) time. For wide
        networks with many more parameters than N·C, use ``KernelLaplace``.
    """

    def __init__(self, model, X, y, likelihood: Likelihood, prior_precision: float, *,
                 param_filter: ParamFilter = None, batch_size: int = 128, max_params: int = 20_000):
        P = flat_params(model, param_filter).numel()
        if P > max_params:
            raise ValueError(
                f"{P} parameters selected; the exact posterior needs ~{8 * P * P / 1e9:.1f} GB. "
                "Use KernelLaplace (cost set by N·C instead of P), param_filter (e.g. last "
                "layer only), or raise max_params.")
        self._setup(model, X, y, likelihood, prior_precision, param_filter, batch_size)
        self.J = jacobians(model, X, param_filter, batch_size)          # (N, C, P), CPU
        N, C, P = self.J.shape

        A = self.alpha * torch.eye(P, dtype=DT)
        for i in range(0, N, batch_size):
            Jb = self.J[i:i + batch_size].to(DT)
            A += Jb.reshape(-1, P).T @ (self.Lam[i:i + batch_size] @ Jb).reshape(-1, P)
        self.L = torch.linalg.cholesky(A)
        self.logdet_A = 2.0 * self.L.diagonal().log().sum().item()
        del A
        self.S = self._blocks(self.J)                                     # J_i Σ J_iᵀ

    def _half_solve(self, B):
        return torch.linalg.solve_triangular(self.L, B, upper=False)

    def _blocks(self, J):
        P, C, bs = self.n_params, J.shape[1], self.batch_size
        S = torch.empty(len(J), C, C, dtype=DT)
        for i in range(0, len(J), bs):
            V = self._half_solve(J[i:i + bs].to(DT).reshape(-1, P).T).reshape(P, -1, C)
            S[i:i + bs] = torch.einsum("pbc,pbd->bcd", V, V)
        return S

    def _set_cov(self, idx):
        V = self._half_solve(self.J[idx].to(DT).reshape(-1, self.n_params).T)
        return V.T @ V

    def _logdet_ratio(self):
        return self.logdet_A - self.n_params * math.log(self.alpha)

    def _new_moments(self, X):
        S = self._blocks(jacobians(self.model, X, self.param_filter, self.batch_size))
        return self._forward(X), S


class KernelLaplace(_Laplace):
    """The same posterior as ``LinearizedLaplace``, computed in output space via the empirical NTK.

    With G = J Jᵀ (N·C × N·C), Λ = blkdiag(Λ_i) and M = αI + Λ^{½} G Λ^{½} (Woodbury),

        J Σ Jᵀ = α⁻¹ [G − G Λ^{½} M⁻¹ Λ^{½} G],     log det(I + GGN/α) = log det(M/α).

    Every per-example, per-set and whole-model quantity is identical to ``LinearizedLaplace``;
    the cost is O((NC)³) time and about 3 (NC)² floats of memory instead of O(P³) and P².
    Use it for wide networks (P ≫ N·C), e.g. MLPs on molecular fingerprints. For networks
    whose parameters are all in ``nn.Linear`` layers, G is assembled without Jacobians
    (see ``capfid.ntk_gram``).

    Parameters are as for ``LinearizedLaplace``, plus

    gram : torch.Tensor, optional
        Precomputed ``ntk_gram(model, X, param_filter=...)``, e.g. to reuse it across several
        prior precisions.
    prior_mean : float or torch.Tensor (C,), optional
        If given, the posterior mean is *refit*: instead of the trained outputs f(x; θ*), it is
        the MAP of the linearized network, i.e. of the Gaussian process with this prior mean
        and kernel G/α (Newton's method; one step for a Gaussian likelihood). Use this when θ*
        is not a stationary point of Σℓ + (α/2)‖θ‖², e.g. after training with dropout or early
        stopping, or with α chosen after training. Unlearning is then exact for a Gaussian
        likelihood (exact Gaussian-process leave-out) instead of relying on ∇L(θ*) = 0, and
        ``totals().log_evidence_nats`` is the (Laplace) Gaussian-process marginal likelihood.
        ``outputs`` and fidelities then refer to this refit mean, not to the network.
    max_outputs : int
        Safety limit on N·C.
    """

    def __init__(self, model, X, y, likelihood: Likelihood, prior_precision: float, *,
                 param_filter: ParamFilter = None, batch_size: int = 512, gram=None,
                 prior_mean=None, max_outputs: int = 20_000, newton_iters: int = 50,
                 newton_tol: float = 1e-10):
        self._setup(model, X, y, likelihood, prior_precision, param_filter, batch_size)
        N, C = self.n, self.n_outputs
        NC = N * C
        if NC > max_outputs:
            raise ValueError(f"N·C = {NC}; the output-space posterior needs ~{24 * NC * NC / 1e9:.1f}"
                             " GB. Raise max_outputs, or use LinearizedLaplace if P is small.")
        G = ntk_gram(model, X, param_filter=param_filter, batch_size=batch_size) if gram is None \
            else gram.detach().cpu().to(DT)
        if G.shape != (NC, NC):
            raise ValueError(f"gram has shape {tuple(G.shape)}, expected ({NC}, {NC})")
        self.X, self.G = X, G
        self.prior_mean = None
        if prior_mean is not None:
            self._refit(prior_mean, newton_iters, newton_tol)
        LhG = torch.einsum("icd,idx->icx", self._Lh, G.reshape(N, C, NC)).reshape(NC, NC)
        M = torch.einsum("xjd,jde->xje", LhG.reshape(NC, N, C), self._Lh).reshape(NC, NC)
        M.diagonal().add_(self.alpha)
        self.L = torch.linalg.cholesky(M)
        del M
        self.logdet_M = 2.0 * self.L.diagonal().log().sum().item()
        self.W = torch.linalg.solve_triangular(self.L, LhG, upper=False)   # L⁻¹ Λ^{½} G
        del LhG
        Gd = torch.diagonal(G.reshape(N, C, N, C), dim1=0, dim2=2).permute(2, 0, 1)
        W3 = self.W.reshape(NC, N, C)
        S = (Gd - torch.einsum("xic,xid->icd", W3, W3)) / self.alpha
        self.S = 0.5 * (S + S.transpose(-1, -2))                           # J_i Σ J_iᵀ

    def _refit(self, prior_mean, iters, tol):
        """Newton's method for the MAP of the Gaussian process (mean m, kernel G/α) in output
        space (Rasmussen & Williams 2006, Alg. 3.1, with block curvature), parametrized as
        f = m + K a and globalized by backtracking on Ψ(a) = log p(y | f) − ½ aᵀ K a: an
        undamped step can overshoot into the saturated region of a sigmoid or softmax when
        K = G/α is large, from which Newton does not recover."""
        N, C = self.n, self.n_outputs
        m = torch.as_tensor(prior_mean, dtype=DT).expand(N, C).clone()
        K = self.G / self.alpha

        def psi(a):
            Ka = K @ a
            f = m + Ka.reshape(N, C)
            return float(self.likelihood.log_lik(f, self.y).sum()) - 0.5 * float(a @ Ka), f

        a = torch.zeros(N * C, dtype=DT)
        obj, f = psi(a)
        for it in range(iters):
            r, Lam = self.likelihood.grad_hess(f, self.y)
            Lh = psd_sqrt(Lam)
            b = (Lam @ (f - m).unsqueeze(-1)).squeeze(-1) - r                        # (N, C)
            Lhm = torch.block_diag(*Lh) if C > 1 else torch.diag(Lh.reshape(-1))
            B = Lhm @ K @ Lhm
            B.diagonal().add_(1.0)
            Lb = torch.linalg.cholesky(B)
            Kb = K @ b.reshape(-1)
            d = b.reshape(-1) - Lhm @ torch.cholesky_solve((Lhm @ Kb).unsqueeze(-1), Lb).squeeze(-1) - a
            t = 1.0
            for _ in range(40):
                obj_new, f_new = psi(a + t * d)
                if obj_new >= obj - 1e-12 * abs(obj):
                    break
                t *= 0.5
            a = a + t * d
            step = float((f_new - f).abs().max())
            f, obj = f_new, obj_new
            if step < tol:
                break
        self.newton_iterations = it + 1
        self.prior_mean = m
        self._a = a                                                     # K⁻¹ (f − m) = −r at the MAP
        self.f = f
        self.r, self.Lam = self.likelihood.grad_hess(f, self.y)
        self._Lh = psd_sqrt(self.Lam)

    def _prior_sq(self):
        if self.prior_mean is None:
            return super()._prior_sq()
        return float(self._a @ self.G @ self._a) / self.alpha ** 2

    def _rows(self, idx):
        C = self.n_outputs
        return (idx.reshape(-1, 1) * C + torch.arange(C)).reshape(-1)

    def _set_cov(self, idx):
        rows = self._rows(idx)
        Wr = self.W[:, rows]
        S = (self.G[rows][:, rows] - Wr.T @ Wr) / self.alpha
        return 0.5 * (S + S.T)

    def _logdet_ratio(self):
        return self.logdet_M - self.n * self.n_outputs * math.log(self.alpha)

    def _new_moments(self, X, chunk_size: int = 1024):
        N, C, NC = self.n, self.n_outputs, self.n * self.n_outputs
        kw = dict(param_filter=self.param_filter, batch_size=self.batch_size)
        out, means = [], []
        for i in range(0, len(X), chunk_size):
            Xb = X[i:i + chunk_size]
            m = len(Xb)
            Gx = ntk_gram(self.model, self.X, Xb, **kw)                     # (NC, mC)
            LhGx = torch.einsum("icd,idx->icx", self._Lh, Gx.reshape(N, C, m * C)).reshape(NC, m * C)
            V = torch.linalg.solve_triangular(self.L, LhGx, upper=False).reshape(NC, m, C)
            Gb = ntk_gram(self.model, Xb, **kw).reshape(m, C, m, C)
            Gd = torch.diagonal(Gb, dim1=0, dim2=2).permute(2, 0, 1)
            S = (Gd - torch.einsum("xic,xid->icd", V, V)) / self.alpha
            out.append(0.5 * (S + S.transpose(-1, -2)))
            if self.prior_mean is not None:                   # GP mean: m + G(x, X) K⁻¹(f − m)/α
                means.append(self.prior_mean[0] + (Gx.T @ self._a).reshape(m, C) / self.alpha)
        f = torch.cat(means) if means else self._forward(X)
        return f, torch.cat(out)
