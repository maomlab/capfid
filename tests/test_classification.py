"""Structural properties for a nonlinear classifier, plus API checks."""
import numpy as np
import pytest
import torch
import torch.nn as nn

import capfid
from capfid.likelihoods import psd_sqrt


def test_leverage_eigenvalues_in_unit_interval(mlp_problem):
    la = mlp_problem["la"]
    Lh = psd_sqrt(la.Lam)
    ev = torch.linalg.eigvalsh(Lh @ la.S @ Lh)
    assert ev.min() > -1e-10 and ev.max() < 1


def test_capacities_nonnegative_and_flips_cost_more(mlp_problem):
    res = mlp_problem["la"].pointwise(n_samples=200)
    for k in ("dof", "volume_bits", "information_bits"):
        assert (res[k] >= -1e-12).all(), k
    flipped = np.zeros(len(res["dof"]), bool)
    flipped[:6] = True
    assert np.median(res["information_bits"][flipped]) > 3 * np.median(res["information_bits"][~flipped])


def test_dof_sum_equals_effective_parameters(mlp_problem):
    la = mlp_problem["la"]
    A = la.L @ la.L.T
    Sigma = torch.linalg.inv(A)
    gamma = torch.trace(Sigma @ (A - la.alpha * torch.eye(la.n_params, dtype=A.dtype))).item()
    assert abs(la.totals().dof - gamma) < 1e-8 * max(1.0, gamma)


def test_singleton_set_matches_pointwise(mlp_problem):
    la = mlp_problem["la"]
    res = la.pointwise(n_samples=10)
    for i in (0, 7, 40):
        s = la.set_capacity([i], n_samples=10)
        assert abs(s["information_bits"] - res["information_bits"][i]) < 1e-9
        assert abs(s["volume_bits"] - res["volume_bits"][i]) < 1e-9
        assert abs(s["dof"] - res["dof"][i]) < 1e-9


def test_joint_volume_at_least_sum_of_singles(mlp_problem):
    """Proposition 3 (submodularity): joint volume >= sum of single-example volumes."""
    la = mlp_problem["la"]
    rng = np.random.default_rng(0)
    for m in (2, 5, 20):
        idx = rng.choice(la.n, m, replace=False)
        s = la.set_capacity(idx, n_samples=10)
        assert s["volume_bits"] >= s["sum_single_volume_bits"] - 1e-9
        assert s["redundancy"] >= 1 - 1e-9
    assert 0 < la.totals().unique_fraction <= 1


def test_totals_identities(mlp_problem):
    t = mlp_problem["la"].totals()
    assert abs(t.description_length_bits + t.log_evidence_nats / np.log(2)) < 1e-8
    assert abs(t.kl_prior_bits - (t.model_bits - 0.5 * t.dof / np.log(2))) < 1e-8


def test_param_filter_last_layer(mlp_problem):
    p = mlp_problem
    la = capfid.LinearizedLaplace(p["model"], p["X"], p["y"], p["lik"], p["alpha"],
                                  param_filter=lambda n: n.startswith("head"))
    assert la.n_params == sum(q.numel() for q in p["model"].head.parameters())
    assert (la.pointwise(n_samples=10)["information_bits"] >= -1e-12).all()


def test_max_params_guard(mlp_problem):
    p = mlp_problem
    with pytest.raises(ValueError):
        capfid.LinearizedLaplace(p["model"], p["X"], p["y"], p["lik"], p["alpha"], max_params=10)


def test_weight_decay_conversion():
    torch.manual_seed(0)
    X, y = torch.randn(32, 3), torch.randint(0, 2, (32,))
    model, lik, wd = nn.Linear(3, 2), capfid.Categorical(), 1e-3
    alpha = capfid.prior_precision_from_weight_decay(wd, len(X))
    params = list(model.parameters())
    g1 = torch.autograd.grad(capfid.map_objective(model, X, y, lik, alpha) / len(X), params)
    g2 = torch.autograd.grad(nn.functional.cross_entropy(model(X), y), params)
    for a, b, q in zip(g1, g2, params):
        torch.testing.assert_close(a, b + wd * q)


def test_gaussian_closed_form_matches_monte_carlo():
    torch.manual_seed(0)
    lik = capfid.Gaussian(0.5)
    m = torch.randn(4, 2, dtype=torch.float64)
    B = torch.randn(4, 2, 2, dtype=torch.float64)
    S = B @ B.transpose(-1, -2) * 0.3
    y = torch.randn(4, 2, dtype=torch.float64)
    eps = torch.randn(200_000, 4, 2, dtype=torch.float64)
    e_cf, p_cf = lik.gauss_expect(m, S, y)
    e_mc, p_mc = capfid.Likelihood.gauss_expect(lik, m, S, y, eps)
    torch.testing.assert_close(e_cf, e_mc, atol=2e-2, rtol=0)
    torch.testing.assert_close(p_cf, p_mc, atol=2e-2, rtol=0)
