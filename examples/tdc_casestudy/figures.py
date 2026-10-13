"""Figures, tables and key-compound reports for the TDC case study.

Run after ``run_casestudy.py all`` from the repository root:

    python examples/tdc_casestudy/figures.py --out paper/figures

Writes figures to --out and tables to examples/tdc_casestudy/results/.
"""
from __future__ import annotations

import argparse
import io
import math
import warnings

import matplotlib
import matplotlib.ticker
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common import CACHE, FEATURES, LN2, RESULTS, TASKS, get_data, load_features, load_task

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7})
FCOL = {"ECFP4": "#4C72B0", "APDP": "#DD8452", "CheMeleon": "#55A868"}
REGIMES = ("early", "long")


# ------------------------------------------------------------------------------ loading
def summary():
    return pd.read_csv(RESULTS / "capfid_summary.csv")


def capfid_table(task, feature, regime="early", seed=0, split="train"):
    return pd.read_csv(CACHE / "capfid" / f"{task}_{feature}_{regime}_seed{seed}_{split}.csv")


def log2_lik(kind, f, y, sigma):
    """log2 p(y | f) for outputs f (logits or standardized means), broadcasting."""
    if kind == "binary":
        return (y * f - np.logaddexp(0, f)) / LN2
    return (-0.5 * (f - y) ** 2 / sigma ** 2 - 0.5 * np.log(2 * np.pi * sigma ** 2)) / LN2


def retraining_memorization(task, feature, regime, sigma):
    """Memorization measured by retraining (Feldman & Zhang 2020): mean log2-likelihood of a
    training compound under models that saw it minus models that did not; also the mean
    change of the output itself (self-influence, in model units)."""
    z = np.load(CACHE / "retrain" / f"{task}_{feature}_{regime}.npz")
    keep, out = z["keep"], z["outputs"]                       # (K, N)
    data = get_data(task, feature)
    y = data.y["train"].numpy().ravel()
    ll = log2_lik(data.kind, out, y[None, :], sigma)
    n_in, n_out = keep.sum(0), (~keep).sum(0)
    mean_in = (ll * keep).sum(0) / np.maximum(n_in, 1)
    mean_out = (ll * ~keep).sum(0) / np.maximum(n_out, 1)
    f_in = (out * keep).sum(0) / np.maximum(n_in, 1)
    f_out = (out * ~keep).sum(0) / np.maximum(n_out, 1)
    return pd.DataFrame(dict(retrain_memorization_bits=mean_in - mean_out,
                             retrain_shift=f_in - f_out, n_in=n_in, n_out=n_out))


# ------------------------------------------------------------------------------- UMAP
def umap_embedding(task, feature):
    """2-D UMAP of all compounds of a task in the given feature space (cached)."""
    path = CACHE / "umap" / f"{task}_{feature}.npy"
    if path.exists():
        return np.load(path)
    import umap

    X = load_features(task, feature)
    if feature == "CheMeleon":
        X = (X - X.mean(0)) / (X.std(0) + 1e-6)
        kw = dict(metric="euclidean")
    else:
        X = X.astype(bool)
        kw = dict(metric="jaccard")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        emb = umap.UMAP(n_neighbors=30, min_dist=0.25, random_state=0, **kw).fit_transform(X)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, emb)
    return emb


# ------------------------------------------------------------------------ chemistry
def mol_image(smiles, size=(260, 200)):
    from rdkit import Chem
    from rdkit.Chem.Draw import rdMolDraw2D

    mol = Chem.MolFromSmiles(smiles)
    d = rdMolDraw2D.MolDraw2DCairo(*size)
    opts = d.drawOptions()
    opts.clearBackground = True
    opts.padding = 0.08
    opts.bondLineWidth = 1.2
    opts.fixedBondLength = 22          # same scale in every panel; large molecules shrink to fit
    rdMolDraw2D.PrepareAndDrawMolecule(d, mol)
    d.FinishDrawing()
    return plt.imread(io.BytesIO(d.GetDrawingText()))


def tanimoto_neighbors(task, query_rows, pool_rows, k=3):
    """k nearest ECFP4 Tanimoto neighbors (excluding the query itself) among ``pool_rows``."""
    X = load_features(task, "ECFP4").astype(bool)
    P = X[pool_rows].astype(np.float32)
    out = {}
    for q in query_rows:
        x = X[q].astype(np.float32)
        inter = P @ x
        sim = inter / (P.sum(1) + x.sum() - inter + 1e-9)
        sim[np.asarray(pool_rows) == q] = -1
        idx = np.argsort(-sim)[:k]
        out[q] = [(int(np.asarray(pool_rows)[i]), float(sim[i])) for i in idx]
    return out


def metric_of(task, y, out, data):
    from common import metric

    if data.kind == "binary":
        return metric("binary", y, out)
    return metric("regression", y, out * data.y_std + data.y_mean)


def per_compound(task):
    """One row per training compound: capfid scores for every featurization and regime,
    memorization measured by retraining, and predictions in the task's units (cached)."""
    path = RESULTS / f"per_compound_{TASKS[task]['short']}.csv.gz"
    if path.exists():
        return pd.read_csv(path)
    S = summary()
    df = load_task(task)
    out = df[df.split == "train"].reset_index().rename(columns={"index": "row"})
    for f in FEATURES:
        data = get_data(task, f)
        for r in REGIMES:
            s0 = S[(S.task == task) & (S.feature == f) & (S.regime == r) & (S.seed == 0)].iloc[0]
            tr = capfid_table(task, f, r)
            mem = retraining_memorization(task, f, r, s0.sigma)
            k = f"{f}_{r}"
            for c in ("dof", "volume_bits", "information_bits", "memorization_bits",
                      "fidelity_bits", "heldout_fidelity_bits", "network_information_bits",
                      "network_memorization_bits"):
                out[f"{c}_{k}"] = tr[c].to_numpy()
            out[f"retrain_memorization_bits_{k}"] = mem.retrain_memorization_bits.to_numpy()
            if data.kind == "binary":
                to = lambda v: 1 / (1 + np.exp(-v))  # noqa: E731
            else:
                to = lambda v: v * data.y_std + data.y_mean  # noqa: E731
            out[f"pred_{k}"] = to(tr.network_output.to_numpy())
            out[f"gp_{k}"] = to(tr.output.to_numpy())
            out[f"gp_heldout_{k}"] = to(tr.heldout_output.to_numpy())
    RESULTS.mkdir(exist_ok=True)
    out.to_csv(path, index=False, float_format="%.6g")
    return out


def model_table():
    """Per task, featurization and regime: performance, Laplace totals, validation."""
    S, M = summary(), pd.read_csv(RESULTS / "test_metrics.csv")
    rows = []
    for t in TASKS:
        pc = per_compound(t)
        for f in FEATURES:
            data = get_data(t, f)
            for r in REGIMES:
                sel = (S.task == t) & (S.feature == f) & (S.regime == r)
                s0 = S[sel & (S.seed == 0)].iloc[0]
                m = M[(M.task == t) & (M.feature == f) & (M.regime == r)]
                te = capfid_table(t, f, r, split="test")
                k = f"{f}_{r}"
                fz = pc[f"retrain_memorization_bits_{k}"]
                rho = lambda a: float(spearmanr(pc[a], fz)[0])  # noqa: E731
                s1 = capfid_table(t, f, r, seed=1)
                err = (np.abs(te.Y - (1 / (1 + np.exp(-te.network_output)))) if data.kind == "binary"
                       else np.abs(te.Y - (te.network_output * data.y_std + data.y_mean)))
                N = int(s0.n_train)
                rows.append(dict(
                    task=TASKS[t]["short"], feature=f, regime=r, epochs=int(m.epochs.iloc[0]),
                    width=int(s0.width), depth=int(s0.depth), dropout=s0.dropout,
                    weight_decay=s0.weight_decay, lr=s0.lr, n_params=int(s0.n_params), n_train=N,
                    train_metric=m.train.mean(), test_metric=m.test.mean(), test_metric_sd=m.test.std(),
                    gp_test_metric=metric_of(t, te.Y, te.output, data),
                    alpha=s0.alpha, sigma=s0.sigma * (data.y_std if data.kind != "binary" else 1),
                    gamma=s0.dof, gamma_per_n=s0.dof / N,
                    volume_total_bits=s0.volume_bits, model_bits=s0.model_bits,
                    data_bits=s0.data_bits, description_bits_per_compound=(s0.model_bits + s0.data_bits) / N,
                    unique_fraction=s0.unique_fraction,
                    info_median=pc[f"information_bits_{k}"].median(),
                    info_p99=pc[f"information_bits_{k}"].quantile(0.99),
                    info_total=pc[f"information_bits_{k}"].sum(),
                    retrain_mem_median=fz.median(), retrain_mem_p99=fz.quantile(0.99),
                    retrain_mem_total=fz.sum(),
                    rho_info_vs_retrain=rho(f"information_bits_{k}"),
                    rho_mem_vs_retrain=rho(f"memorization_bits_{k}"),
                    rho_network_info_vs_retrain=rho(f"network_information_bits_{k}"),
                    rho_network_mem_vs_retrain=rho(f"network_memorization_bits_{k}"),
                    network_mem_max=pc[f"network_memorization_bits_{k}"].max(),
                    rho_seed01=float(spearmanr(pc[f"information_bits_{k}"], s1.information_bits)[0]),
                    rho_alpha_x0p1=s0["spearman_info_alpha_x0.1"],
                    rho_alpha_x10=s0["spearman_info_alpha_x10"],
                    rho_testvol_vs_error=float(spearmanr(te.volume_bits, err)[0]),
                    refit_vs_network_rmse=s0.refit_vs_network_rmse))
    T = pd.DataFrame(rows)
    T.to_csv(RESULTS / "models.csv", index=False, float_format="%.6g")
    return T


def percentile_rank(x):
    return pd.Series(x).rank(pct=True).to_numpy()


def key_compounds(task, regime="early", n_consensus=4):
    """Compounds that a capacity analysis would flag, chosen by fixed rules on the
    information each model stores (refit posterior, seed 0):

    consensus     highest mean information percentile across the three featurizations
    specific      for each featurization, the compound in its top 1% whose percentile most
                  exceeds the larger of the other two (representation-specific capacity)
    atypical      highest mean volume percentile among compounds with below-median information
                  (capacity allotted to an unusual input whose label the model already predicts)
    """
    pc = per_compound(task)
    P = {f: percentile_rank(pc[f"information_bits_{f}_{regime}"]) for f in FEATURES}
    V = np.mean([percentile_rank(pc[f"volume_bits_{f}_{regime}"]) for f in FEATURES], 0)
    mean_p = np.mean([P[f] for f in FEATURES], 0)
    chosen, why = [], []
    for i in np.argsort(-mean_p)[:n_consensus]:
        chosen.append(int(i)), why.append("consensus")
    for f in FEATURES:
        others = np.max([P[g] for g in FEATURES if g != f], 0)
        score = np.where(P[f] >= 0.99, P[f] - others, -np.inf)
        score[chosen] = -np.inf
        i = int(np.argmax(score))
        if np.isfinite(score[i]):
            chosen.append(i), why.append(f"{f}-specific")
    low = mean_p < 0.5
    score = np.where(low, V, -np.inf)
    score[chosen] = -np.inf
    chosen.append(int(np.argmax(score))), why.append("atypical")
    out = pc.iloc[chosen].copy()
    out.insert(0, "reason", why)
    out.insert(0, "key", np.arange(1, len(out) + 1))
    out["mean_info_percentile"] = mean_p[chosen]
    out["mean_volume_percentile"] = V[chosen]
    nn = tanimoto_neighbors(task, out.row.tolist(), pc.row.tolist(), k=3)
    df = load_task(task)
    for j in range(3):
        out[f"nn{j + 1}_id"] = [df.Drug_ID[nn[q][j][0]] for q in out.row]
        out[f"nn{j + 1}_sim"] = [nn[q][j][1] for q in out.row]
        out[f"nn{j + 1}_y"] = [df.Y[nn[q][j][0]] for q in out.row]
    return out


def _fmt_y(task, v):
    if TASKS[task]["kind"] == "binary":
        return "pos" if v > 0.5 else "neg"
    return f"{v:.2f}"


def _fmt_pred(task, v):
    return f"p={v:.2f}" if TASKS[task]["kind"] == "binary" else f"{v:.2f}"


def _short(text, n):
    text = str(text)
    return text if len(text) <= n else text[:n - 1] + "\u2026"


def fig_task(task, keys, out, regime="early"):
    """UMAP of all compounds in each feature space, training compounds colored by the
    information the corresponding model stores; key compounds numbered and depicted."""
    from matplotlib.colors import LogNorm

    pc = per_compound(task)
    df = load_task(task)
    n_keys = len(keys)
    ncol = 4
    nrow = int(math.ceil(n_keys / ncol))
    fig = plt.figure(figsize=(7.0, 2.5 + 1.95 * nrow))
    gs = fig.add_gridspec(1 + nrow, 12, height_ratios=[1.3] + [1.0] * nrow, hspace=0.08,
                          wspace=0.3)
    norm = LogNorm(1e-3, 30)
    tr_rows = pc.row.to_numpy()
    for j, f in enumerate(FEATURES):
        ax = fig.add_subplot(gs[0, 4 * j:4 * j + 4])
        emb = umap_embedding(task, f)
        info = pc[f"information_bits_{f}_{regime}"].clip(1e-3, 30).to_numpy()
        other = np.setdiff1d(np.arange(len(df)), tr_rows)
        ax.scatter(*emb[other].T, s=1.5, c="0.85", lw=0, rasterized=True)
        order = np.argsort(info)
        sc = ax.scatter(*emb[tr_rows[order]].T, s=2.0, c=info[order], cmap="magma_r", norm=norm,
                        lw=0, rasterized=True)
        for _, k in keys.iterrows():
            x, y = emb[int(k.row)]
            ax.scatter([x], [y], s=40, facecolors="none", edgecolors="#1f77b4", lw=1.0)
            ax.annotate(str(int(k.key)), (x, y), xytext=(3, 3), textcoords="offset points",
                        fontsize=7, color="#1f77b4", fontweight="bold")
        ax.set_title(f"{f}", fontsize=8)
        ax.set_xticks([]), ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    cax = fig.add_axes([0.92, 0.70, 0.008, 0.18])
    cb = fig.colorbar(sc, cax=cax)
    cb.set_label("information (bits)", fontsize=6.5)
    cb.ax.tick_params(labelsize=6)

    for n, (_, k) in enumerate(keys.iterrows()):
        r, c = divmod(n, ncol)
        sub = gs[1 + r, 3 * c:3 * c + 3].subgridspec(2, 1, height_ratios=[1.0, 0.55], hspace=0.0)
        ax = fig.add_subplot(sub[0])
        ax.imshow(mol_image(df.Drug[int(k.row)]))
        ax.axis("off")
        ax.set_title(f"{int(k.key)}  {_short(k.Drug_ID, 25)}", fontsize=6.3, loc="left", pad=1,
                     fontweight="bold")
        tx = fig.add_subplot(sub[1])
        tx.axis("off")
        info = "/".join(f"{k[f'information_bits_{f}_{regime}']:.2g}" for f in FEATURES)
        preds = "/".join(_fmt_pred(task, k[f"pred_{f}_{regime}"]).replace("p=", "")
                         for f in FEATURES)
        txt = (f"label {_fmt_y(task, k.Y)};  pred {preds}\n"
               f"$\\mathcal{{I}}$ = {info} bits\n"
               f"nearest (Tc {k.nn1_sim:.2f}): {_fmt_y(task, k.nn1_y)}\n"
               f"{k.reason}")
        tx.text(0.02, 1.0, txt, transform=tx.transAxes, fontsize=5.6, va="top", ha="left",
                linespacing=1.25)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_validation(out):
    """capfid estimates versus memorization measured by retraining, for every task (rows),
    with the refit posterior in both training regimes and the unrefit one-step estimate."""
    fig, axes = plt.subplots(3, 3, figsize=(7.0, 6.3))
    cols = [("information_bits", "early", "refit $\\mathcal{I}$, early stopping"),
            ("information_bits", "long", "refit $\\mathcal{I}$, 100 epochs"),
            ("network_memorization_bits", "long", "one-step memorization from $\\theta^*$, 100 epochs")]
    for i, t in enumerate(TASKS):
        pc = per_compound(t)
        for j, (c, r, title) in enumerate(cols):
            ax = axes[i, j]
            one_step = c.startswith("network")
            for f in FEATURES:
                x = pc[f"{c}_{f}_{r}"]
                y = pc[f"retrain_memorization_bits_{f}_{r}"]
                rho = spearmanr(x, y)[0]
                if not one_step:
                    x = x.clip(1e-4, None)
                ax.scatter(x, y, s=1.2, lw=0, alpha=0.2, color=FCOL[f], rasterized=True)
                q = pd.qcut(x.rank(method="first"), 12, labels=False)
                g = pd.DataFrame(dict(x=x, y=y)).groupby(q)
                ax.plot(g.x.median(), g.y.median(), "o-", ms=2.5, lw=1.2, color=FCOL[f],
                        mec="white", mew=0.3, label=f"{f} ρ={rho:.2f}")
            if one_step:                      # can be negative away from a stationary point
                ax.set_xscale("symlog", linthresh=0.01)
                ax.set_xticks([t for t in (0, 0.01, 1, 100) if t <= ax.get_xlim()[1]])
                ax.axvline(0, color="0.6", lw=0.5)
            else:
                ax.set_xscale("log")
                # one tick per decade, so the 1e-4 clipping floor is always labeled
                ax.xaxis.set_major_locator(matplotlib.ticker.LogLocator(numticks=10))
            ax.set_yscale("symlog", linthresh=0.1)
            ax.axhline(0, color="0.6", lw=0.5)
            ax.legend(loc="lower right", handlelength=1.2, handletextpad=0.3, frameon=True,
                      framealpha=0.85, edgecolor="none", fontsize=6)
            if i == 0:
                ax.set_title(title)
            if i == 2:
                ax.set_xlabel("capfid estimate (bits)")
            if j == 0:
                ax.set_ylabel(f"{TASKS[t]['short']}: retraining\nmemorization (bits)")
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)


def test_errors(task, feature, regime="early"):
    """Test compounds with prospective volume and the network's absolute error."""
    data = get_data(task, feature)
    te = capfid_table(task, feature, regime, split="test")
    if data.kind == "binary":
        pred = 1 / (1 + np.exp(-te.network_output))
    else:
        pred = te.network_output * data.y_std + data.y_mean
    te["abs_error"] = np.abs(te.Y - pred)
    return te


def fig_generalization(out):
    """Top: test error against the prospective volume of each test compound (label-free).
    Bottom: distribution of stored information per training compound, early versus long."""
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.3))
    for j, t in enumerate(TASKS):
        ax = axes[0, j]
        for f in FEATURES:
            te = test_errors(t, f)
            q = pd.qcut(te.volume_bits.rank(method="first"), 5, labels=False)
            g = te.groupby(q)
            rho = spearmanr(te.volume_bits, te.abs_error)[0]
            ax.plot(g.volume_bits.median(), g.abs_error.mean(), "o-", ms=3, color=FCOL[f],
                    label=f"{f} ρ={rho:.2f}")
        ax.set_xscale("log")
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_title(TASKS[t]["short"])
        ax.set_xlabel("prospective volume (bits)")
        ax.set_ylabel("mean |error|" + ("" if TASKS[t]["kind"] == "binary"
                                         else f" ({TASKS[t]['unit']})"))
        ax.legend(frameon=False, fontsize=6)
        ax = axes[1, j]
        pc = per_compound(t)
        for f in FEATURES:
            for r, ls in (("early", "-"), ("long", "--")):
                x = np.sort(pc[f"information_bits_{f}_{r}"].clip(1e-4, None))
                ax.plot(x, 1 - np.arange(len(x)) / len(x), ls, color=FCOL[f], lw=1,
                        label=f"{f} {r}" if j == 0 else None)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(1 / len(pc), 1)
        ax.set_xlabel("information $\\mathcal{I}(\\{i\\})$ (bits)")
        ax.set_ylabel("fraction of compounds ≥ x")
    axes[1, 0].legend(frameon=False, fontsize=5.5, ncol=1)
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)


def collisions_table():
    """Groups of training compounds that a featurization cannot tell apart (identical
    vectors; early-stopped models, seed 0). A group conflicts if its labels differ (AMES) or
    span more than 1 log unit (regression). ``code_bits`` is the members' share of the
    model's data code length, -Σ log2 p(y | refit mean)."""
    G = pd.read_csv(RESULTS / "duplicate_groups.csv")
    G = G[(G.regime == "early") & (G.kind == "features")]
    S = summary()
    rows = []
    for t in TASKS:
        for f in FEATURES:
            g = G[(G.task == t) & (G.feature == f)]
            s0 = S[(S.task == t) & (S.feature == f) & (S.regime == "early") & (S.seed == 0)].iloc[0]
            span = g.y_max - g.y_min
            c = g[span > (0.5 if TASKS[t]["kind"] == "binary" else 1.0)]
            big = g.loc[g.n.idxmax()] if len(g) else None
            rows.append(dict(
                task=TASKS[t]["short"], feature=f, n_train=int(s0.n_train), groups=len(g),
                compounds=int(g.n.sum()), conflicting_groups=len(c),
                conflicting_compounds=int(c.n.sum()),
                conflicting_info_joint=c.information_bits.sum(),
                conflicting_info_singles=c.sum_single_information_bits.sum(),
                conflicting_code_bits=-c.fidelity_bits.sum(),
                conflicting_code_share=-c.fidelity_bits.sum() / s0.data_bits,
                largest_n=int(big.n) if big is not None else 0,
                largest_span=float(big.y_max - big.y_min) if big is not None else np.nan,
                largest_dof=float(big.dof) if big is not None else np.nan,
                largest_info=float(big.information_bits) if big is not None else np.nan,
                largest_code_bits=float(-big.fidelity_bits) if big is not None else np.nan))
    T = pd.DataFrame(rows)
    T.to_csv(RESULTS / "collisions.csv", index=False, float_format="%.4g")
    return T


def latex_table(T):
    """Rows for the manuscript's model table."""
    lines = []
    for _, r in T.iterrows():
        lines.append(" & ".join([
            r.task if r.feature == "ECFP4" and r.regime == "early" else "",
            r.feature if r.regime == "early" else "", r.regime, str(r.epochs),
            f"{r.test_metric:.3f}", f"{r.gp_test_metric:.3f}",
            f"{r.gamma:.0f}", f"{r.description_bits_per_compound:.2f}",
            f"{r.info_total:.0f}", f"{r.retrain_mem_total:,.0f}".replace(",", "{,}"),
            f"{r.rho_info_vs_retrain:.2f}", f"{r.rho_network_mem_vs_retrain:.2f}",
            f"{r.rho_seed01:.2f}"]) + r" \\")
    return "\n".join(lines)


def main(out):
    import pathlib

    out = pathlib.Path(out)
    out.mkdir(parents=True, exist_ok=True)
    T = model_table()
    with pd.option_context("display.width", 250, "display.max_columns", 60):
        print(T.round(3).to_string())
    print(latex_table(T))
    with pd.option_context("display.width", 250, "display.max_columns", 60):
        print(collisions_table().round(3).to_string())
    fig_validation(out / "fig4_tdc_validation.png")
    fig_generalization(out / "fig8_tdc_generalization.png")
    for i, t in enumerate(TASKS):
        short = TASKS[t]["short"]
        keys = key_compounds(t)
        keys.to_csv(RESULTS / f"key_compounds_{short}.csv", index=False, float_format="%.4g")
        fig_task(t, keys, out / f"fig{5 + i}_tdc_{short.lower()}.png")
        print(f"\n== {short} key compounds ==")
        cols = ["key", "reason", "Drug_ID", "Drug", "Y"] + \
               [f"information_bits_{f}_early" for f in FEATURES] + ["nn1_sim", "nn1_y"]
        print(keys[cols].to_string(index=False))


def main_umap():
    for t in TASKS:
        for f in FEATURES:
            umap_embedding(t, f)
            print("umap", t, f, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--umap-only", action="store_true")
    ap.add_argument("--out", default="figures")
    args = ap.parse_args()
    if args.umap_only:
        main_umap()
    else:
        main(args.out)
