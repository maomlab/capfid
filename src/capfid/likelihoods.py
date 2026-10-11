"""Output-space likelihoods.

Each likelihood supplies, at the network outputs f (logits or means):

* ``log_lik(f, y)``       log p(y | f), broadcasting over leading sample dims
* ``grad_hess(f, y)``     residual r = ∂ℓ/∂f and curvature Λ = ∂²ℓ/∂f², ℓ = −log p
* ``hess(f)``             Λ alone; it does not depend on y for the likelihoods here
* ``gauss_expect``        E_{f~N(m,S)} log p(y|f)  and  log E_{f~N(m,S)} p(y|f), per example
* ``joint_expect``        the same two quantities for a set whose outputs are jointly Gaussian
"""
from __future__ import annotations

import math

import torch

__all__ = ["Likelihood", "Categorical", "Bernoulli", "Gaussian", "psd_sqrt"]


def psd_sqrt(M: torch.Tensor) -> torch.Tensor:
    """Symmetric PSD square root of a (batch of) matrix ``(..., C, C)``."""
    e, U = torch.linalg.eigh(0.5 * (M + M.transpose(-1, -2)))
    return (U * e.clamp_min(0).sqrt().unsqueeze(-2)) @ U.transpose(-1, -2)


class Likelihood:
    """Base class. Subclasses implement ``log_lik`` and ``grad_hess``."""

    def log_lik(self, f: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def nll(self, f: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return -self.log_lik(f, y)

    def grad_hess(self, f: torch.Tensor, y: torch.Tensor):
        raise NotImplementedError

    def hess(self, f: torch.Tensor) -> torch.Tensor:
        """Curvature Λ at outputs f (needed for unlabeled inputs, e.g. ``predict``)."""
        raise NotImplementedError

    def gauss_expect(self, mean, cov, y, eps):
        """Monte Carlo default. mean (N,C), cov (N,C,C), eps (M,N,C) standard normals."""
        f = mean + torch.einsum("ncd,mnd->mnc", psd_sqrt(cov), eps)
        lp = self.log_lik(f, y)                                     # (M, N)
        return lp.mean(0), torch.logsumexp(lp, 0) - math.log(lp.shape[0])

    def joint_expect(self, mean, cov, y, eps):
        """Monte Carlo default. mean (m,C), cov (mC,mC), eps (M, mC). Returns scalars."""
        m, C = mean.shape
        e, U = torch.linalg.eigh(0.5 * (cov + cov.T))
        f = (mean.reshape(-1) + eps @ (U * e.clamp_min(0).sqrt()).T).reshape(-1, m, C)
        lp = self.log_lik(f, y).sum(-1)                             # (M,)
        return lp.mean(), torch.logsumexp(lp, 0) - math.log(lp.shape[0])


class Categorical(Likelihood):
    """Softmax / cross-entropy. ``y`` is a LongTensor of class indices, shape (N,)."""

    def log_lik(self, f, y):
        return f.log_softmax(-1).gather(-1, y.expand(f.shape[:-1]).unsqueeze(-1)).squeeze(-1)

    def grad_hess(self, f, y):
        p = f.softmax(-1)
        r = p.clone()
        r[torch.arange(len(y)), y] -= 1.0
        return r, self.hess(f)

    def hess(self, f):
        p = f.softmax(-1)
        return torch.diag_embed(p) - p.unsqueeze(-1) * p.unsqueeze(-2)


class Bernoulli(Likelihood):
    """Independent sigmoid outputs (binary cross-entropy). ``y`` holds 0/1 targets shaped like
    the outputs, (N, D); D = 1 for a single binary task with one logit."""

    def log_lik(self, f, y):
        return (y * f - torch.nn.functional.softplus(f)).sum(-1)

    def grad_hess(self, f, y):
        return f.sigmoid() - y, self.hess(f)

    def hess(self, f):
        p = f.sigmoid()
        return torch.diag_embed(p * (1 - p))


class Gaussian(Likelihood):
    """Homoscedastic Gaussian with fixed noise ``sigma``. ``y`` has the shape of the outputs, (N, D).

    Expectations are computed in closed form (no Monte Carlo).
    """

    def __init__(self, sigma: float):
        self.s2 = float(sigma) ** 2

    def log_lik(self, f, y):
        D = f.shape[-1]
        return -0.5 * ((f - y) ** 2).sum(-1) / self.s2 - 0.5 * D * math.log(2 * math.pi * self.s2)

    def grad_hess(self, f, y):
        return (f - y) / self.s2, self.hess(f)

    def hess(self, f):
        D = f.shape[-1]
        return (torch.eye(D, dtype=f.dtype, device=f.device).expand(f.shape[0], D, D) / self.s2).clone()

    def _gauss_logpdf(self, d, cov):
        D = d.shape[-1]
        L = torch.linalg.cholesky(cov + self.s2 * torch.eye(D, dtype=cov.dtype, device=cov.device))
        z = torch.linalg.solve_triangular(L, d.unsqueeze(-1), upper=False).squeeze(-1)
        return (-0.5 * (z ** 2).sum(-1) - L.diagonal(dim1=-2, dim2=-1).log().sum(-1)
                - 0.5 * D * math.log(2 * math.pi))

    def gauss_expect(self, mean, cov, y, eps=None):
        D = mean.shape[-1]
        d = y - mean
        e_ll = (-0.5 * ((d ** 2).sum(-1) + cov.diagonal(dim1=-2, dim2=-1).sum(-1)) / self.s2
                - 0.5 * D * math.log(2 * math.pi * self.s2))
        return e_ll, self._gauss_logpdf(d, cov)

    def joint_expect(self, mean, cov, y, eps=None):
        m, D = mean.shape
        d = (y - mean).reshape(-1)
        e_ll = (-0.5 * ((d ** 2).sum() + cov.diagonal().sum()) / self.s2
                - 0.5 * m * D * math.log(2 * math.pi * self.s2))
        return e_ll, self._gauss_logpdf(d, cov)
