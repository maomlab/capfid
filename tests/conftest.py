import math

import pytest
import torch
import torch.nn as nn

import capfid


class MLP(nn.Module):
    def __init__(self, d_in=2, h=8, c=3):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(d_in, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh())
        self.head = nn.Linear(h, c)

    def forward(self, x):
        return self.head(self.body(x))


@pytest.fixture(scope="session")
def linear_problem():
    """Bayesian linear regression on random Fourier features: Laplace is exact here."""
    g = torch.Generator().manual_seed(0)
    N, D, sigma, alpha = 60, 12, 0.3, 1.0
    x = torch.rand(N, 1, generator=g, dtype=torch.float64) * 6 - 3
    W = torch.randn(1, D, generator=g, dtype=torch.float64) * 1.5
    b = torch.rand(D, generator=g, dtype=torch.float64) * 2 * math.pi
    Phi = torch.cos(x @ W + b) * math.sqrt(2 / D)
    y = torch.sin(2 * x) + sigma * torch.randn(N, 1, generator=g, dtype=torch.float64)
    y[:3] += 2.0
    A = Phi.T @ Phi / sigma ** 2 + alpha * torch.eye(D, dtype=torch.float64)
    w = torch.linalg.solve(A, Phi.T @ y / sigma ** 2)
    model = nn.Linear(D, 1, bias=False).double()
    model.weight.data = w.T.clone()
    lik = capfid.Gaussian(sigma)
    la = capfid.LinearizedLaplace(model, Phi, y, lik, alpha)
    return dict(Phi=Phi, y=y, A=A, w=w, sigma=sigma, alpha=alpha, model=model, lik=lik, la=la)


@pytest.fixture(scope="session")
def mlp_problem():
    """Small 3-class MLP with a few flipped labels, trained to a MAP estimate."""
    g = torch.Generator().manual_seed(1)
    N = 90
    X = torch.randn(N, 2, generator=g)
    y = (torch.atan2(X[:, 1], X[:, 0]) / (2 * math.pi / 3)).floor().long() % 3
    y[:6] = (y[:6] + 1) % 3
    torch.manual_seed(0)
    model = MLP()
    lik, alpha = capfid.Categorical(), 0.5
    capfid.train_map(model, X, y, lik, alpha, adam_steps=500, lr=1e-2, lbfgs_iters=300)
    la = capfid.LinearizedLaplace(model, X, y, lik, alpha)
    return dict(X=X, y=y, model=model, lik=lik, alpha=alpha, la=la)
