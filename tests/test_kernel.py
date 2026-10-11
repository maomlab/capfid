"""KernelLaplace (output space, empirical NTK) must reproduce LinearizedLaplace exactly.

Models are converted to float64 so that the comparisons are not limited by float32 rounding."""
import copy

import numpy as np
import pytest
import torch
import torch.nn as nn

import capfid

def _double(p):
    return copy.deepcopy(p["model"]).double(), p["X"].double()


KEYS = ("dof", "volume_bits", "information_bits", "shift_bits", "fidelity_bits",
        "heldout_fidelity_bits", "memorization_bits")


def _compare(la, ka, idx_sets, X_new=None, y_new=None, tol=1e-8):
    a, b = la.pointwise(n_samples=50), ka.pointwise(n_samples=50)
    for k in KEYS + ("expected_fidelity_bits", "predictive_fidelity_bits"):
        np.testing.assert_allclose(b[k], a[k], atol=tol, rtol=1e-7, err_msg=k)
    ta, tb = la.totals().as_dict(), ka.totals().as_dict()
    for k in ta:
        assert abs(ta[k] - tb[k]) <= tol * max(1.0, abs(ta[k])), k
    for idx in idx_sets:
        sa, sb = la.set_capacity(idx, n_samples=50), ka.set_capacity(idx, n_samples=50)
        for k in sa:
            assert abs(sa[k] - sb[k]) <= tol * max(1.0, abs(sa[k])), k
    if X_new is not None:
        pa, pb = la.predict(X_new, y_new, n_samples=50), ka.predict(X_new, y_new, n_samples=50)
        for k in pa:
            np.testing.assert_allclose(pb[k], pa[k], atol=tol, rtol=1e-7, err_msg=k)


def test_kernel_matches_parameter_space_categorical(mlp_problem):
    p = mlp_problem
    model, X = _double(p)
    la = capfid.LinearizedLaplace(model, X, p["y"], p["lik"], p["alpha"])
    ka = capfid.KernelLaplace(model, X, p["y"], p["lik"], p["alpha"])
    X_new = torch.randn(7, 2, generator=torch.Generator().manual_seed(3), dtype=torch.float64)
    _compare(la, ka, [[0, 1, 2, 3, 4, 5], [10, 50, 70]], X_new, torch.arange(7) % 3)


def test_kernel_matches_parameter_space_gaussian(linear_problem):
    p = linear_problem
    ka = capfid.KernelLaplace(p["model"], p["Phi"], p["y"], p["lik"], p["alpha"])
    _compare(p["la"], ka, [[0, 1, 2], list(range(10, 30))], p["Phi"][:5], p["y"][:5], tol=1e-9)


def test_kernel_matches_parameter_space_bernoulli():
    g = torch.Generator().manual_seed(2)
    X = torch.randn(80, 3, generator=g)
    y = ((X[:, 0] * X[:, 1] > 0).float()).unsqueeze(1)
    y[:5] = 1 - y[:5]
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(3, 16), nn.ReLU(), nn.Dropout(0.2), nn.Linear(16, 16),
                          nn.Tanh(), nn.Linear(16, 1))
    lik, alpha = capfid.Bernoulli(), 0.5
    capfid.train_map(model, X, y, lik, alpha, adam_steps=300, lr=1e-2, lbfgs_iters=100)
    model, X, y = model.double(), X.double(), y.double()
    la = capfid.LinearizedLaplace(model, X, y, lik, alpha)
    ka = capfid.KernelLaplace(model, X, y, lik, alpha)
    _compare(la, ka, [[0, 1, 2, 3, 4], [20, 21]], X[:9] + 0.1, y[:9])


@pytest.mark.parametrize("filt", [None, lambda n: n.startswith("head")])
def test_linear_gram_matches_jacobian_gram(mlp_problem, filt):
    model, X = _double(mlp_problem)
    for a, b in [(X, None), (X, X[:13] * 1.5)]:
        g1 = capfid.ntk_gram(model, a, b, param_filter=filt, method="linear")
        g2 = capfid.ntk_gram(model, a, b, param_filter=filt, method="jacobian")
        torch.testing.assert_close(g1, g2, atol=1e-12, rtol=1e-10)


def test_linear_gram_rejects_nonlinear_parameters():
    model = nn.Sequential(nn.Linear(3, 4), nn.LayerNorm(4), nn.Linear(4, 2))
    X = torch.randn(5, 3)
    with pytest.raises(ValueError):
        capfid.ntk_gram(model, X, method="linear")
    torch.testing.assert_close(capfid.ntk_gram(model, X),                 # auto falls back
                               capfid.ntk_gram(model, X, method="jacobian"))


def test_kernel_last_layer_filter(mlp_problem):
    p = mlp_problem
    model, X = _double(p)
    head = lambda n: n.startswith("head")  # noqa: E731
    la = capfid.LinearizedLaplace(model, X, p["y"], p["lik"], p["alpha"], param_filter=head)
    ka = capfid.KernelLaplace(model, X, p["y"], p["lik"], p["alpha"], param_filter=head)
    _compare(la, ka, [[0, 3, 9]])


def test_bernoulli_derivatives_match_autograd():
    f = torch.randn(6, 2, dtype=torch.float64, requires_grad=True)
    y = torch.randint(0, 2, (6, 2)).to(torch.float64)
    lik = capfid.Bernoulli()
    r, Lam = lik.grad_hess(f.detach(), y)
    g = torch.autograd.grad(lik.nll(f, y).sum(), f)[0]
    torch.testing.assert_close(r, g)
    for i in range(6):
        H = torch.autograd.functional.hessian(lambda z: lik.nll(z, y[i]), f[i].detach())
        torch.testing.assert_close(Lam[i], H)


def test_predict_volume_is_added_point_volume(linear_problem):
    """Adding x to the training data and then removing it must give back predict's volume."""
    p = linear_problem
    Phi, s2 = p["Phi"], p["sigma"] ** 2
    x_new = Phi[:1] * 0.7 + Phi[1:2] * 0.3
    vol = p["la"].predict(x_new)["volume_bits"][0]
    A = p["A"] + x_new.T @ x_new / s2
    exact = 0.5 * (torch.logdet(A) - torch.logdet(p["A"])).item() / np.log(2)
    assert abs(vol - exact) < 1e-12


# ------------------------------------------------------------------ refit (prior_mean) mode
def test_refit_is_identity_at_a_map_estimate(linear_problem):
    """A bias-free linear model at its MAP has function-space prior mean 0: refitting with
    prior_mean=0 must change nothing."""
    p = linear_problem
    ka = capfid.KernelLaplace(p["model"], p["Phi"], p["y"], p["lik"], p["alpha"], prior_mean=0.0)
    np.testing.assert_allclose(ka.f.numpy(), p["la"].f.numpy(), atol=1e-10)
    _compare(p["la"], ka, [[0, 1, 2]], p["Phi"][:5], p["y"][:5], tol=1e-9)


def _unconverged_regressor():
    g = torch.Generator().manual_seed(4)
    X = torch.randn(50, 2, generator=g, dtype=torch.float64)
    y = (torch.sin(2 * X[:, :1]) + 0.2 * torch.randn(50, 1, generator=g, dtype=torch.float64))
    y[:3] += 2.0
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(2, 16), nn.Tanh(), nn.Linear(16, 1)).double()
    capfid.train_map(model, X, y, capfid.Gaussian(0.3), 1.0, adam_steps=50, lr=1e-2, lbfgs_iters=0)
    return model, X, y


def test_refit_gaussian_is_exact_gaussian_process():
    """With a Gaussian likelihood the refit posterior is a GP with kernel G/α: leave-one-out
    predictions, information (Proposition 2) and evidence must match closed forms."""
    model, X, y = _unconverged_regressor()
    sigma, alpha, m = 0.3, 2.0, 0.1
    ka = capfid.KernelLaplace(model, X, y, capfid.Gaussian(sigma), alpha, prior_mean=m)
    K = capfid.ntk_gram(model, X) / alpha
    A = K + sigma ** 2 * torch.eye(len(X), dtype=torch.float64)
    Ai = torch.linalg.inv(A)
    d = (y - m).ravel()
    np.testing.assert_allclose(ka.f.numpy().ravel(), (m + K @ Ai @ d).numpy(), atol=1e-10)
    mu_loo = y.ravel() - (Ai @ d) / Ai.diagonal()
    var_loo = 1 / Ai.diagonal()
    ll_loo = -0.5 * (y.ravel() - mu_loo) ** 2 / var_loo - 0.5 * torch.log(2 * np.pi * var_loo)
    res = ka.pointwise(return_logits=True)
    np.testing.assert_allclose(res["heldout_predictive_bits"], (ll_loo / np.log(2)).numpy(), atol=1e-9)
    np.testing.assert_allclose(res["heldout_outputs"].ravel(), (mu_loo).numpy(), atol=1e-9)
    gap = res["expected_fidelity_bits"] - res["heldout_predictive_bits"]
    np.testing.assert_allclose(res["information_bits"], gap, atol=1e-9)
    log_ml = (-0.5 * d @ Ai @ d - 0.5 * torch.logdet(A) - 0.5 * len(X) * np.log(2 * np.pi)).item()
    assert abs(ka.totals().log_evidence_nats - log_ml) < 1e-8
    # predictions at the training inputs reproduce the refit mean
    np.testing.assert_allclose(ka.predict(X)["outputs"], ka.f.numpy(), atol=1e-9)


def test_refit_bernoulli_reaches_stationary_point():
    g = torch.Generator().manual_seed(5)
    X = torch.randn(60, 3, generator=g, dtype=torch.float64)
    y = (X[:, :1] > 0).double()
    y[:4] = 1 - y[:4]
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(3, 12), nn.ReLU(), nn.Linear(12, 1)).double()
    alpha, m = 0.7, -0.2
    ka = capfid.KernelLaplace(model, X, y, capfid.Bernoulli(), alpha, prior_mean=m)
    G = capfid.ntk_gram(model, X)
    r = (ka.f.sigmoid() - y).ravel()
    np.testing.assert_allclose((ka.f.ravel() - m).numpy(), (-G @ r / alpha).numpy(), atol=1e-9)
    # Laplace marginal likelihood of GP classification (Rasmussen & Williams eq. 3.32)
    W = (ka.f.sigmoid() * (1 - ka.f.sigmoid())).ravel()
    B = torch.eye(len(X), dtype=torch.float64) + W.sqrt()[:, None] * G / alpha * W.sqrt()[None, :]
    a = -r
    log_q = (capfid.Bernoulli().log_lik(ka.f, y).sum() - 0.5 * a @ G @ a / alpha
             - 0.5 * torch.logdet(B)).item()
    assert abs(ka.totals().log_evidence_nats - log_q) < 1e-8


@pytest.mark.parametrize("alpha, m", [(1e-2, 12.0), (1e-3, 12.0), (0.7, 20.0)])
def test_refit_bernoulli_converges_from_a_saturated_start(alpha, m):
    # an undamped Newton step from a saturated prior mean overshoots to |f| ~ 10^3-10^5
    g = torch.Generator().manual_seed(5)
    X = torch.randn(60, 3, generator=g, dtype=torch.float64)
    y = (X[:, :1] > 0).double()
    y[:4] = 1 - y[:4]
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(3, 12), nn.ReLU(), nn.Linear(12, 1)).double()
    ka = capfid.KernelLaplace(model, X, y, capfid.Bernoulli(), alpha, prior_mean=m)
    G = capfid.ntk_gram(model, X)
    r = (ka.f.sigmoid() - y).ravel()
    assert ka.f.abs().max() < 30
    np.testing.assert_allclose((ka.f.ravel() - m).numpy(), (-G @ r / alpha).numpy(), atol=1e-6)
