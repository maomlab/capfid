"""TDC case study: train dropout MLPs on three ADMET tasks and three featurizations, then ask
capfid which training compounds the models spend capacity on.

Stages (each caches its results under cache/ and can be rerun independently):

  hpo       random search (24 configurations) on the validation split, early stopping
  train     final models with the best configuration in two regimes, 3 seeds each:
            "early" stops at the validation-selected epoch, "long" trains for 100 epochs;
            test metrics
  capfid    output-space Laplace posterior of each trained network (capfid.KernelLaplace),
            with the mean refit to the MAP of the linearized network (a Gaussian process
            with the network's tangent kernel) and the prior precision (and Gaussian noise)
            set by the training-set marginal likelihood (noise: validation split, with
            the prior precision profiled out);
            per-example capacity and fidelity for every training compound; test-set
            predictions
  groups    joint capacity of groups of training compounds with identical feature vectors
            or the same constitution (capfid set_capacity)
  retrain   memorization measured by retraining: 48 models on random 70% subsets
            (Feldman & Zhang 2020), compared with capfid's one-shot estimates
  figures   see figures.py

Run from the repository root (about 2.5 hours on 10 CPU cores):

    python examples/tdc_casestudy/run_casestudy.py all --workers 10
    python examples/tdc_casestudy/figures.py --out paper/figures
"""
from __future__ import annotations

import argparse
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import torch

from common import (CACHE, FEATURES, LN2, MLP, RESULTS, TASKS, HParams, evaluate, get_data,
                    predict, search_space, train_mlp)

N_CONFIGS, SEEDS, N_RETRAIN, RETRAIN_FRAC = 24, (0, 1, 2), 48, 0.7
REGIMES = ("early", "long")
LONG_EPOCHS = 100
CAPFID_SEEDS = {"early": (0, 1, 2), "long": (0, 1)}


def _init_worker():
    torch.set_num_threads(1)


def _pool(workers):
    return ProcessPoolExecutor(workers, initializer=_init_worker)


def model_path(task, feature, regime, seed=0):
    return CACHE / "models" / f"{task}_{feature}_{regime}_seed{seed}.pt"


# ------------------------------------------------------------------------------- hpo
def _hpo_job(args):
    task, feature, k, hp = args
    data = get_data(task, feature)
    t0 = time.time()
    _, epoch, valid = train_mlp(data, hp, seed=0, early_stopping=True)
    return dict(task=task, feature=feature, config=k, **hp.as_dict(), best_epoch=epoch,
                valid=valid, seconds=time.time() - t0)


def stage_hpo(workers):
    configs = search_space(N_CONFIGS)
    jobs = [(t, f, k, hp) for t in TASKS for f in FEATURES for k, hp in enumerate(configs)
            if not (CACHE / "hpo" / f"{t}_{f}.csv").exists()]
    if not jobs:
        return
    # longest jobs first keeps the pool busy
    jobs.sort(key=lambda j: -j[3].width * j[3].depth / j[3].lr)
    rows = []
    with _pool(workers) as ex:
        for i, r in enumerate(ex.map(_hpo_job, jobs)):
            rows.append(r)
            print(f"  hpo {i + 1}/{len(jobs)} {r['task']} {r['feature']} cfg{r['config']} "
                  f"valid={r['valid']:.4f} epoch={r['best_epoch']} {r['seconds']:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    for (t, f), g in df.groupby(["task", "feature"]):
        path = CACHE / "hpo" / f"{t}_{f}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        g.sort_values("config").to_csv(path, index=False)


def best_hparams(task, feature, regime="early") -> HParams:
    """Best validation configuration; "early" trains for its selected number of epochs,
    "long" for LONG_EPOCHS."""
    df = pd.read_csv(CACHE / "hpo" / f"{task}_{feature}.csv")
    kind = TASKS[task]["kind"]
    r = df.loc[df.valid.idxmax() if kind == "binary" else df.valid.idxmin()]
    return HParams(int(r.width), int(r.depth), float(r.dropout), float(r.weight_decay),
                   float(r.lr), epochs=int(r.best_epoch) if regime == "early" else LONG_EPOCHS)


# ----------------------------------------------------------------------------- train
def _train_job(args):
    task, feature, regime, seed = args
    data = get_data(task, feature)
    hp = best_hparams(task, feature, regime)
    model, _, _ = train_mlp(data, hp, seed=seed)
    path = model_path(task, feature, regime, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    return dict(task=task, feature=feature, regime=regime, seed=seed, epochs=hp.epochs,
                train=evaluate(model, data, "train"), valid=evaluate(model, data, "valid"),
                test=evaluate(model, data, "test"))


def stage_train(workers):
    jobs = [(t, f, r, s) for t in TASKS for f in FEATURES for r in REGIMES for s in SEEDS]
    jobs.sort(key=lambda j: j[2] != "long")
    with _pool(workers) as ex:
        rows = list(ex.map(_train_job, jobs))
    df = pd.DataFrame(rows)
    RESULTS.mkdir(exist_ok=True)
    df.to_csv(RESULTS / "test_metrics.csv", index=False)
    print(df.groupby(["task", "feature", "regime"])[["train", "valid", "test"]]
          .agg(["mean", "std"]).round(4))


def load_model(task, feature, regime="early", seed=0):
    data = get_data(task, feature)
    hp = best_hparams(task, feature, regime)
    model = MLP(data.X["train"].shape[1], hp.width, hp.depth, hp.dropout)
    model.load_state_dict(torch.load(model_path(task, feature, regime, seed)))
    return model.eval(), data, hp


# ---------------------------------------------------------------------------- capfid
def prior_mean(data):
    """Function-space prior mean: the training-set average (0 for standardized targets; the
    log-odds of the positive rate for a binary task)."""
    if data.kind == "binary":
        p = float(data.y["train"].mean())
        return math.log(p / (1 - p))
    return 0.0


class TangentGP:
    """The refit linearized network viewed as a Gaussian process with kernel G/α and constant
    prior mean m, used only to choose α and the Gaussian noise σ (Rasmussen & Williams 2006,
    Alg. 3.1-3.2, eqs. 2.25-2.26, 2.30, 3.21-3.24, 3.32).

    The prior precision α maximizes the training-set marginal likelihood (Laplace, for
    classification). The noise σ maximizes the predictive likelihood of held-out compounds,
    with α profiled out: a kernel fit to the training labels can explain their noise as
    signal, so the training-set marginal likelihood underestimates σ."""

    def __init__(self, G, y, m, kind):
        self.G, self.y, self.m, self.kind = G, y.ravel(), m, kind
        if kind != "binary":
            d, self.U = torch.linalg.eigh(G)
            self.d = d.clamp_min(0)
            self.z = self.U.T @ (self.y - m)
        self._a, self._alpha = torch.zeros_like(self.y), 1.0
        self._B = {}

    # -- classification: Newton mode at the training inputs
    def _mode(self, alpha):
        """f = m + K a maximizing Ψ(a) = log p(y | f) − ½ aᵀ K a, by Newton's method with
        backtracking. Warm-started from the previous mode's f (rescaled a) unless a = 0 is
        better: an undamped step from a mode found at a much larger α overshoots into the
        saturated region of the sigmoid and never recovers."""
        K, m, y = self.G / alpha, self.m, self.y

        def psi(a):
            Ka = K @ a
            f = m + Ka
            return float((y * f - torch.nn.functional.softplus(f)).sum() - 0.5 * a @ Ka), f

        a = self._a * (alpha / self._alpha)
        obj, f = psi(a)
        obj0, f0 = psi(torch.zeros_like(a))
        if not obj >= obj0:
            a, obj, f = torch.zeros_like(a), obj0, f0
        for _ in range(100):
            p = torch.sigmoid(f)
            w = (p * (1 - p)).sqrt()
            B = w[:, None] * K * w[None, :]
            B.diagonal().add_(1.0)
            Lb = torch.linalg.cholesky(B)
            b = w ** 2 * (f - m) + (y - p)
            d = b - w * torch.cholesky_solve((w * (K @ b)).unsqueeze(-1), Lb).squeeze(-1) - a
            t = 1.0
            for _ in range(40):
                obj_new, f_new = psi(a + t * d)
                if obj_new >= obj - 1e-12 * abs(obj):
                    break
                t *= 0.5
            a = a + t * d
            done = float((f_new - f).abs().max()) < 1e-9
            f, obj = f_new, obj_new
            if done:
                break
        self._a, self._alpha = a, alpha
        p = torch.sigmoid(f)
        w = (p * (1 - p)).sqrt()
        B = w[:, None] * K * w[None, :]
        B.diagonal().add_(1.0)
        return f, y - p, w, torch.linalg.cholesky(B)

    def log_ml(self, alpha, sigma=1.0):
        if self.kind == "binary":
            f, a, _, Lb = self._mode(alpha)
            loglik = float((self.y * f - torch.nn.functional.softplus(f)).sum())
            return loglik - 0.5 * float(a @ self.G @ a) / alpha - float(Lb.diagonal().log().sum())
        c = self.d / alpha + sigma ** 2
        return float(-0.5 * (self.z ** 2 / c + torch.log(2 * math.pi * c)).sum())

    def predictive(self, alpha, sigma, Gxt, gxx):
        """Predictive mean and variance of the latent function at held-out inputs."""
        if self.kind == "binary":
            f, a, w, Lb = self._mode(alpha)
            mu = self.m + Gxt @ a / alpha
            v = torch.linalg.solve_triangular(Lb, (w[:, None] * Gxt.T) / alpha, upper=False)
            return mu, (gxx / alpha - (v ** 2).sum(0)).clamp_min(0)
        key = id(Gxt)
        if key not in self._B:
            self._B[key] = Gxt @ self.U
        B, tau = self._B[key], alpha * sigma ** 2
        mu = self.m + B @ (self.z / (self.d + tau))
        return mu, ((gxx - (B ** 2 / (self.d + tau)).sum(1)) / alpha).clamp_min(0)

    def heldout_loglik(self, alpha, sigma, Gxt, gxx, yx):
        """Total log predictive likelihood (nats) of held-out labels; probit approximation
        for classification."""
        mu, var = self.predictive(alpha, sigma, Gxt, gxx)
        yx = yx.ravel()
        if self.kind == "binary":
            z = mu / torch.sqrt(1 + math.pi * var / 8)
            return float((yx * z - torch.nn.functional.softplus(z)).sum())
        v = var + sigma ** 2
        return float(-0.5 * ((yx - mu) ** 2 / v + torch.log(2 * math.pi * v)).sum())

    def alpha_given_sigma(self, sigma):
        """α maximizing the training-set marginal likelihood at noise σ."""
        from scipy.optimize import minimize_scalar

        r = minimize_scalar(lambda la: -self.log_ml(math.exp(la), sigma), bounds=(-8, 14),
                            method="bounded", options=dict(xatol=1e-5))
        return math.exp(r.x)

    def fit_profile(self, Gxt, gxx, yx):
        """α by marginal likelihood; σ (regression) by held-out predictive likelihood."""
        from scipy.optimize import minimize_scalar

        if self.kind == "binary":
            return self.alpha_given_sigma(1.0), 1.0
        r = minimize_scalar(
            lambda ls: -self.heldout_loglik(self.alpha_given_sigma(math.exp(ls)), math.exp(ls),
                                            Gxt, gxx, yx),
            bounds=(-5, 1.5), method="bounded", options=dict(xatol=1e-4))
        sigma = math.exp(r.x)
        return self.alpha_given_sigma(sigma), sigma

    def fit(self, objective):
        """Maximize ``objective(alpha, sigma)`` over log α (and log σ)."""
        from scipy.optimize import minimize, minimize_scalar

        if self.kind == "binary":
            r = minimize_scalar(lambda la: -objective(math.exp(la), 1.0), bounds=(-6, 12),
                                method="bounded", options=dict(xatol=1e-4))
            return math.exp(r.x), 1.0
        best = min((minimize(lambda v: -objective(math.exp(v[0]), math.exp(v[1])), x0,
                             method="Nelder-Mead", options=dict(xatol=1e-6, fatol=1e-8,
                                                                maxiter=4000))
                    for x0 in ([0.0, -1.0], [4.0, -0.5], [8.0, -2.0], [2.0, 0.0])),
                   key=lambda r: r.fun)
        return math.exp(best.x[0]), math.exp(best.x[1])


def _likelihood(kind, sigma):
    import capfid

    return capfid.Bernoulli() if kind == "binary" else capfid.Gaussian(sigma)


def _capfid_job(args):
    """Refit Laplace posterior of one trained network, with α and σ chosen by
    ``TangentGP.fit_profile`` on the validation split. Hyperparameters that maximize the
    training-set marginal likelihood jointly over α and σ are evaluated for comparison."""
    import capfid
    from scipy.stats import spearmanr

    task, feature, regime, seed = args
    torch.set_num_threads(int(os.environ.get("CAPFID_THREADS", "4")))
    t0 = time.time()
    model, data, hp = load_model(task, feature, regime, seed)
    X, y = data.X["train"], data.y["train"].double()
    G = capfid.ntk_gram(model, X)
    held = {}
    for split in ("valid", "test"):
        Xs = data.X[split]
        held[split] = (capfid.ntk_gram(model, Xs, X), capfid.ntk_gram(model, Xs).diagonal(),
                       data.y[split].double())
    m = prior_mean(data)
    gp = TangentGP(G, y, m, data.kind)
    alpha, sigma = gp.fit_profile(*held["valid"])
    alpha_ev, sigma_ev = (alpha, sigma) if data.kind == "binary" else gp.fit(gp.log_ml)
    diag = {}
    for name, (a, s_) in (("val", (alpha, sigma)), ("ev", (alpha_ev, sigma_ev))):
        nt = len(held["test"][2])
        nv = len(held["valid"][2])
        diag[f"test_bits_{name}"] = gp.heldout_loglik(a, s_, *held["test"]) / nt / LN2
        diag[f"valid_bits_{name}"] = gp.heldout_loglik(a, s_, *held["valid"]) / nv / LN2
        diag[f"train_log_ml_{name}"] = gp.log_ml(a, s_)
        if data.kind != "binary":
            mu, var = gp.predictive(a, s_, *held["test"][:2])
            z = (held["test"][2].ravel() - mu) / torch.sqrt(var + s_ ** 2)
            diag[f"test_z_sd_{name}"] = float(z.std())
    t_hyper = time.time() - t0

    lik = _likelihood(data.kind, sigma)
    la = capfid.KernelLaplace(model, X, y, lik, alpha, gram=G, prior_mean=m)
    pw = la.pointwise(n_samples=1000, return_logits=True)
    tot = la.totals().as_dict()
    seconds = time.time() - t0

    # the same posterior without refitting: centered on the trained network (assumes ∇L(θ*) = 0)
    net = capfid.KernelLaplace(model, X, y, lik, alpha, gram=G).pointwise(n_samples=10,
                                                                          return_logits=True)
    f_net = predict(model, X).double()
    summary = dict(task=task, feature=feature, regime=regime, seed=seed, n_train=len(X),
                   alpha=alpha, sigma=sigma, alpha_evidence=alpha_ev, sigma_evidence=sigma_ev,
                   prior_mean=m, seconds=seconds, seconds_hyperparameters=t_hyper,
                   newton_iterations=la.newton_iterations, **hp.as_dict(), **tot, **diag,
                   alpha_weight_decay=len(X) * hp.weight_decay / sigma ** 2,
                   refit_vs_network_rmse=float(((la.f - f_net) ** 2).mean().sqrt()))
    table = pd.DataFrame({k: v for k, v in pw.items() if v.ndim == 1})
    if seed != 0:
        return summary, table, None

    for fac in (0.1, 10.0):              # sensitivity of the ranking to the prior precision
        alt = capfid.KernelLaplace(model, X, y, lik, alpha * fac, gram=G,
                                   prior_mean=m).pointwise(n_samples=10)
        summary[f"spearman_info_alpha_x{fac:g}"] = float(
            spearmanr(alt["information_bits"], pw["information_bits"])[0])

    df = data.df[data.df.split == "train"].reset_index().rename(columns={"index": "row"})
    table["output"] = la.f.numpy().ravel()                     # refit (tangent-GP) mean
    table["heldout_output"] = pw["heldout_outputs"].ravel()
    table["output_var"] = la.S.numpy().ravel()
    table["network_output"] = f_net.numpy().ravel()
    for k in ("information_bits", "memorization_bits", "fidelity_bits", "heldout_fidelity_bits"):
        table[f"network_{k}"] = net[k]
    table = pd.concat([df, table], axis=1)

    yt = data.y["test"].double()
    pt = la.predict(data.X["test"], yt, n_samples=1000)
    dt = data.df[data.df.split == "test"].reset_index().rename(columns={"index": "row"})
    test = pd.concat([dt, pd.DataFrame(dict(
        output=pt["outputs"].ravel(), output_var=pt["output_cov"].ravel(),
        network_output=predict(model, data.X["test"]).numpy().ravel(),
        volume_bits=pt["volume_bits"], fidelity_bits=pt["fidelity_bits"],
        predictive_fidelity_bits=pt["predictive_fidelity_bits"]))], axis=1)
    return summary, table, test


def stage_capfid(workers, tasks=TASKS):
    jobs = [(t, f, r, s) for t in tasks for f in FEATURES for r in REGIMES
            for s in CAPFID_SEEDS[r]]
    summaries = []
    with ProcessPoolExecutor(workers) as ex:
        for (t, f, r, s), (summary, table, test) in zip(jobs, ex.map(_capfid_job, jobs)):
            summaries.append(summary)
            base = CACHE / "capfid" / f"{t}_{f}_{r}_seed{s}"
            base.parent.mkdir(parents=True, exist_ok=True)
            table.to_csv(f"{base}_train.csv", index=False)
            if test is not None:
                test.to_csv(f"{base}_test.csv", index=False)
            print(f"  capfid {t} {f} {r} seed{s}: alpha={summary['alpha']:.3g} "
                  f"sigma={summary['sigma']:.3g} (evidence: {summary['alpha_evidence']:.3g}, "
                  f"{summary['sigma_evidence']:.3g}) gamma={summary['dof']:.1f} "
                  f"test bits/cpd {summary['test_bits_val']:.3f} vs {summary['test_bits_ev']:.3f} "
                  f"{summary['seconds']:.0f}s", flush=True)
    RESULTS.mkdir(exist_ok=True)
    out, new = RESULTS / "capfid_summary.csv", pd.DataFrame(summaries)
    if set(tasks) != set(TASKS) and out.exists():            # rerun of some tasks: merge
        old = pd.read_csv(out)
        new = pd.concat([old[~old.task.isin(tasks)], new], ignore_index=True)
        new["_order"] = new.task.map(list(TASKS).index)
        new = new.sort_values(["_order"], kind="stable").drop(columns="_order")
    new.to_csv(out, index=False)


# ---------------------------------------------------------------------------- groups
def duplicate_groups(data):
    """Groups (size ≥ 2) of training compounds that the model cannot tell apart (identical
    feature vectors) or that are the same compound (shared InChIKey skeleton, i.e. the same
    constitution up to stereochemistry and protonation)."""
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    X = data.X["train"].numpy()
    _, inv = np.unique(X, axis=0, return_inverse=True)
    smiles = data.df.Drug[data.df.split == "train"]
    skel = np.array([Chem.MolToInchiKey(Chem.MolFromSmiles(s)).split("-")[0] for s in smiles])
    out = []
    for kind, labels in (("features", inv.ravel()), ("skeleton", skel)):
        for lab in pd.unique(labels):
            idx = np.flatnonzero(labels == lab)
            if len(idx) > 1:
                out.append((kind, idx))
    return out


def _groups_job(args):
    """Joint capacity (``set_capacity``) of every duplicate group, seed-0 posterior."""
    import capfid

    task, feature, regime = args
    torch.set_num_threads(int(os.environ.get("CAPFID_THREADS", "4")))
    model, data, _ = load_model(task, feature, regime, 0)
    S = pd.read_csv(RESULTS / "capfid_summary.csv")
    s0 = S[(S.task == task) & (S.feature == feature) & (S.regime == regime) & (S.seed == 0)].iloc[0]
    la = capfid.KernelLaplace(model, data.X["train"], data.y["train"].double(),
                              _likelihood(data.kind, s0.sigma), s0.alpha,
                              prior_mean=prior_mean(data))
    y = data.y_raw("train")
    rows = []
    for g, (kind, idx) in enumerate(duplicate_groups(data)):
        r = la.set_capacity(torch.as_tensor(idx), n_samples=200)
        rows.append(dict(task=task, feature=feature, regime=regime, group=g, kind=kind,
                         members=";".join(map(str, idx)), y_min=y[idx].min(), y_max=y[idx].max(),
                         y_mean=y[idx].mean(), **r))
    return pd.DataFrame(rows)


def stage_groups(workers):
    jobs = [(t, f, r) for t in TASKS for f in FEATURES for r in REGIMES]
    with ProcessPoolExecutor(workers) as ex:
        tables = list(ex.map(_groups_job, jobs))
    T = pd.concat(tables, ignore_index=True)
    T.to_csv(RESULTS / "duplicate_groups.csv", index=False, float_format="%.6g")
    print(T.groupby(["task", "feature", "regime", "kind"]).agg(
        groups=("n", "size"), compounds=("n", "sum"), info=("information_bits", "sum"),
        info_singles=("sum_single_information_bits", "sum")).round(1).to_string())


# --------------------------------------------------------------------------- retrain
def _retrain_job(args):
    task, feature, regime, k = args
    data = get_data(task, feature)
    hp = best_hparams(task, feature, regime)
    n = len(data.X["train"])
    keep = np.random.default_rng(1000 + k).random(n) < RETRAIN_FRAC
    model, _, _ = train_mlp(data, hp, seed=1000 + k, keep=keep)
    return task, feature, regime, k, keep, predict(model, data.X["train"]).numpy().ravel()


def retrain_path(task, feature, regime):
    return CACHE / "retrain" / f"{task}_{feature}_{regime}.npz"


def stage_retrain(workers):
    """Each model trains on a random 70% of the training set with a new seed; for every
    training compound we keep its output from models that did and did not see it."""
    jobs = [(t, f, r, k) for t in TASKS for f in FEATURES for r in REGIMES
            for k in range(N_RETRAIN) if not retrain_path(t, f, r).exists()]
    if not jobs:
        return
    jobs.sort(key=lambda j: j[2] != "long")
    res = {}
    with _pool(workers) as ex:
        for i, (t, f, r, k, keep, out) in enumerate(ex.map(_retrain_job, jobs)):
            res.setdefault((t, f, r), {})[k] = (keep, out)
            if (i + 1) % 48 == 0:
                print(f"  retrain {i + 1}/{len(jobs)}", flush=True)
    for (t, f, r), d in res.items():
        ks = sorted(d)
        path = retrain_path(t, f, r)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, keep=np.stack([d[k][0] for k in ks]),
                            outputs=np.stack([d[k][1] for k in ks]))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["hpo", "train", "capfid", "groups", "retrain", "all"])
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--tasks", nargs="+", choices=list(TASKS), default=list(TASKS),
                    help="capfid stage only: rerun these tasks and merge the summary")
    args = ap.parse_args()
    stages = (["hpo", "train", "capfid", "groups", "retrain"] if args.stage == "all"
              else [args.stage])
    run = dict(hpo=stage_hpo, train=stage_train, retrain=stage_retrain,
               capfid=lambda w: stage_capfid(max(1, w // 4), args.tasks),
               groups=lambda w: stage_groups(max(1, w // 4)))       # 4 BLAS threads per job
    for s in stages:
        t0 = time.time()
        print(f"== {s} ==", flush=True)
        run[s](args.workers)
        print(f"== {s} done in {time.time() - t0:.0f}s ==", flush=True)


if __name__ == "__main__":
    main()
