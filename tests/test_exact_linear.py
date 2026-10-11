"""In Bayesian linear regression every formula must match brute-force refits exactly."""
import math

import numpy as np
import torch

import capfid

LN2 = math.log(2)


def _brute_force(p):
    Phi, y, A, w, s2 = p["Phi"], p["y"], p["A"], p["w"], p["sigma"] ** 2
    N, D = Phi.shape
    Sig = torch.linalg.inv(A)
    kl, ll_loo = np.zeros(N), np.zeros(N)
    for i in range(N):
        keep = torch.arange(N) != i
        Ai = A - Phi[i:i + 1].T @ Phi[i:i + 1] / s2
        wi = torch.linalg.solve(Ai, Phi[keep].T @ y[keep] / s2)
        d = wi - w
        kl[i] = 0.5 * (torch.trace(Ai @ Sig) - D + (d.T @ Ai @ d).item()
                       - torch.logdet(Ai) - torch.logdet(Sig)).item() / LN2
        mu, v = (Phi[i] @ wi).item(), s2 + (Phi[i] @ torch.linalg.solve(Ai, Phi[i])).item()
        ll_loo[i] = (-0.5 * (y[i, 0].item() - mu) ** 2 / v - 0.5 * math.log(2 * math.pi * v)) / LN2
    return kl, ll_loo


def test_information_and_heldout_match_refits(linear_problem):
    res = linear_problem["la"].pointwise()
    kl, ll_loo = _brute_force(linear_problem)
    np.testing.assert_allclose(res["information_bits"], kl, atol=1e-10)
    np.testing.assert_allclose(res["heldout_predictive_bits"], ll_loo, atol=1e-10)


def test_bayes_identity(linear_problem):
    """Proposition 2: information = expected train fit − held-out predictive fit."""
    res = linear_problem["la"].pointwise()
    gap = res["expected_fidelity_bits"] - res["heldout_predictive_bits"]
    np.testing.assert_allclose(res["information_bits"], gap, atol=1e-10)


def test_volume_is_leverage_formula(linear_problem):
    """Corollary 1: volume = −½ log2(1 − h_ii); Σ dof = tr(hat matrix)."""
    p = linear_problem
    Phi, a, s2 = p["Phi"], p["alpha"], p["sigma"] ** 2
    eye = torch.eye(Phi.shape[1], dtype=Phi.dtype)
    h = (Phi @ torch.linalg.solve(Phi.T @ Phi + a * s2 * eye, Phi.T)).diagonal().numpy()
    res = p["la"].pointwise()
    np.testing.assert_allclose(res["volume_bits"], -0.5 * np.log2(1 - h), atol=1e-12)
    np.testing.assert_allclose(res["dof"].sum(), h.sum(), atol=1e-10)


def test_newton_step_is_exact_for_linear_model(linear_problem):
    p = linear_problem
    res = p["la"].pointwise(return_logits=True)
    for i in (0, 10, 30):
        m2 = capfid.retrain_without(p["model"], p["Phi"], p["y"], p["lik"], p["alpha"], [i])
        with torch.no_grad():
            f_new = m2(p["Phi"][i:i + 1]).item()
        assert abs(f_new - res["heldout_outputs"][i, 0]) < 1e-6


def test_singleton_set_matches_pointwise(linear_problem):
    la = linear_problem["la"]
    res = la.pointwise()
    for i in (0, 5):
        s = la.set_capacity([i])
        assert abs(s["information_bits"] - res["information_bits"][i]) < 1e-10
        assert abs(s["volume_bits"] - res["volume_bits"][i]) < 1e-10
        assert abs(s["heldout_predictive_bits"] - res["heldout_predictive_bits"][i]) < 1e-10
