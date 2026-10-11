"""
reproduce_paper.py — regenerate every figure and number in the capfid manuscript.

  0. Bayesian linear regression: Laplace is exact here, so every formula is checked
     against brute-force leave-one-out refits.
  1. Two moons + 10% flipped labels, small tanh MLP, prior precision α swept:
     the compression knob. Watch flipped points move from "ignored" to "memorized".
  2. Tiny Vision Transformer on sklearn 8x8 digits + 5% flipped labels:
     per-example scatter, highest-capacity images, and set (per-class) capacity.
  3. Validation: Newton/Laplace leave-one-out vs actual warm-started retraining,
     and the Bayes identity  KL = ll_train − ll_loo.

Run from the repository root:

    pip install -e ".[examples]"
    python examples/paper/reproduce_paper.py --out paper/figures      # ~12 min, 1 CPU core
    python examples/paper/reproduce_paper.py --quick --out /tmp/figs   # ~1 min smoke test

Numbers are printed to stdout; figures are written to --out.
"""
import argparse
import math
import os
import time

import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits, make_moons

from capfid import Categorical, Gaussian, LinearizedLaplace, retrain_without, train_map

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

LN2 = math.log(2.0)
OUT = "figures"
# full settings reproduce the manuscript; --quick shrinks everything for a smoke test
CFG = dict(moons_adam=3000, moons_lbfgs=1000, vit_adam=3000, vit_lbfgs=500, vit_n=1000,
           val_mlp=(20, 20), val_vit=(12, 16), retrain_iters=300)
QUICK = dict(moons_adam=300, moons_lbfgs=100, vit_adam=200, vit_lbfgs=30, vit_n=300,
             val_mlp=(3, 3), val_vit=(2, 2), retrain_iters=30)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
CLEAN, FLIP = "#4C72B0", "#D62728"


def tic():
    t = time.time()
    return lambda msg: print(f"  [{time.time() - t:6.1f}s] {msg}", flush=True)


# ============================================================================ #
# 0. Exact sanity check: Bayesian linear regression on random Fourier features
# ============================================================================ #
def sanity_linear():
    print("\n== 0. Bayesian linear regression (exact check) ==")
    torch.manual_seed(0)
    N, D, sigma, alpha = 200, 40, 0.3, 1.0
    x = torch.rand(N, 1, dtype=torch.float64) * 6 - 3
    W, b = torch.randn(1, D, dtype=torch.float64) * 1.5, torch.rand(D, dtype=torch.float64) * 2 * math.pi
    Phi = torch.cos(x @ W + b) * math.sqrt(2 / D)
    y = torch.sin(2 * x) + sigma * torch.randn(N, 1, dtype=torch.float64)
    y[:5] += 2.0                                                 # a few outliers

    A = Phi.T @ Phi / sigma ** 2 + alpha * torch.eye(D, dtype=torch.float64)
    w = torch.linalg.solve(A, Phi.T @ y / sigma ** 2)
    model = nn.Linear(D, 1, bias=False).double()
    model.weight.data = w.T.clone()
    la = LinearizedLaplace(model, Phi, y, Gaussian(sigma), alpha)
    pw = la.pointwise()

    # brute force: exact posterior without point i
    kl_ex, ll_loo_ex = np.zeros(N), np.zeros(N)
    Sig = torch.linalg.inv(A)
    for i in range(N):
        keep = torch.arange(N) != i
        Ai = A - Phi[i:i + 1].T @ Phi[i:i + 1] / sigma ** 2
        wi = torch.linalg.solve(Ai, Phi[keep].T @ y[keep] / sigma ** 2)
        Sig_i = torch.linalg.inv(Ai)
        d = wi - w
        kl = 0.5 * (torch.trace(Ai @ Sig) - D + (d.T @ Ai @ d).item()
                    + torch.logdet(Ai.inverse()) - torch.logdet(Sig))
        kl_ex[i] = kl.item() / LN2
        mu, v = (Phi[i] @ wi).item(), sigma ** 2 + (Phi[i] @ Sig_i @ Phi[i]).item()
        ll_loo_ex[i] = (-0.5 * (y[i, 0].item() - mu) ** 2 / v - 0.5 * math.log(2 * math.pi * v)) / LN2
    hat = Phi @ torch.linalg.solve(Phi.T @ Phi + alpha * sigma ** 2 * torch.eye(D, dtype=torch.float64), Phi.T)
    lev = hat.diagonal().numpy()

    print(f"  max |KL_laplace − KL_exact|              = {np.abs(pw['information_bits'] - kl_ex).max():.2e} bits")
    print(f"  max |ll_loo_laplace − ll_loo_exact|      = {np.abs(pw['heldout_predictive_bits'] - ll_loo_ex).max():.2e} bits")
    print(f"  max |KL − (ll_train − ll_loo)|           = "
          f"{np.abs(pw['information_bits'] - (pw['expected_fidelity_bits'] - pw['heldout_predictive_bits'])).max():.2e} bits")
    print(f"  max |entropy − (−½log2(1−h_ii))|          = {np.abs(pw['volume_bits'] + 0.5 * np.log2(1 - lev)).max():.2e} bits")
    print(f"  Σ dof_i = {pw['dof'].sum():.3f}  = tr(hat) = {lev.sum():.3f}  (effective # params of {D})")
    print(f"  outliers: mean KL {pw['information_bits'][:5].mean():.2f} bits vs inliers {pw['information_bits'][5:].mean():.3f} bits")


# ============================================================================ #
# 1. Two moons + MLP, sweeping prior precision α
# ============================================================================ #
class MLP(nn.Module):
    def __init__(self, d_in=2, h=32, c=2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh(), nn.Linear(h, c))

    def forward(self, x):
        return self.net(x)


def moons(n=300, flip=0.10, seed=0):
    X, y = make_moons(n, noise=0.2, random_state=seed)
    flipped = np.random.default_rng(seed).random(n) < flip
    y[flipped] = 1 - y[flipped]
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y), flipped


HO_CMAP, HO_MIN = "viridis", -8.0


def capacity_scatter(ax, pw, flipped, xlim=(1e-4, 30), s=12):
    """x: capacity (KL bits); y: fidelity log2 p(y|x,θ*); color: held-out log2 p(y|x,θ*_-i)."""
    kl = np.clip(pw["information_bits"], *xlim)
    ll = pw["fidelity_bits"]
    ho = np.clip(pw["heldout_fidelity_bits"], HO_MIN, 0)
    kw = dict(cmap=HO_CMAP, vmin=HO_MIN, vmax=0)
    sc = ax.scatter(kl[~flipped], ll[~flipped], s=s, c=ho[~flipped], lw=0, alpha=0.85, **kw)
    ax.scatter(kl[flipped], ll[flipped], s=s * 2.4, c=ho[flipped], marker="D", edgecolors=FLIP,
               linewidths=1.1, **kw)
    ax.set_xscale("log")
    ax.set_xlim(*xlim)
    ax.set_ylim(min(-3.0, ll.min() - 0.3), 0.3)
    ax.set_xlabel("capacity: KL(q_D ‖ q_D∖i)  [bits]")
    ax.set_ylabel("fidelity: log₂ p(yᵢ | xᵢ, θ*)  [bits]")
    return sc


def scatter_legend(ax, loc="lower left"):
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", ls="", color="0.5", label="clean label"),
         Line2D([], [], marker="D", ls="", mfc="0.85", mec=FLIP, mew=1.1, label="flipped label")]
    ax.legend(handles=h, loc=loc, fontsize=7, frameon=False)


def add_colorbar(fig, sc, ax):
    cb = fig.colorbar(sc, ax=ax, fraction=0.05, pad=0.02)
    cb.set_label("held-out log₂ p(yᵢ | xᵢ, θ*₋ᵢ)  [bits]", fontsize=7.5)
    cb.ax.tick_params(labelsize=7)


def demo_moons(alphas=(3.0, 0.2, 0.03)):
    print("\n== 1. Two moons MLP, prior-precision sweep ==")
    X, y, flipped = moons()
    lik = Categorical()
    fig, axes = plt.subplots(2, len(alphas), figsize=(4.2 * len(alphas), 7.6))
    gx, gy = np.meshgrid(np.linspace(-1.8, 2.8, 200), np.linspace(-1.4, 1.9, 200))
    G = torch.tensor(np.c_[gx.ravel(), gy.ravel()], dtype=torch.float32)
    keep = {}
    for j, a in enumerate(alphas):
        t = tic()
        torch.manual_seed(1)
        model = train_map(MLP(), X, y, lik, a, adam_steps=CFG["moons_adam"], lr=1e-2, lbfgs_iters=CFG["moons_lbfgs"])
        la = LinearizedLaplace(model, X, y, lik, a)
        pw = la.pointwise()
        tot = la.totals()
        pred = la.f.argmax(-1).numpy()
        mem = (pred[flipped] == y.numpy()[flipped]).mean()
        t(f"α={a}: P={tot.n_params}, dof γ={tot.dof:.1f}, entropy drop={tot.volume_bits:.0f} bits, "
          f"model={tot.model_bits:.0f} + data={tot.data_bits:.0f} bits, "
          f"log evidence={tot.log_evidence_nats:.1f} nats, flipped fit={mem:.0%}, "
          f"KL flipped/clean median={np.median(pw['information_bits'][flipped]):.3f}/"
          f"{np.median(pw['information_bits'][~flipped]):.4f} bits, "
          f"Σ unique entropy / total = {pw['volume_bits'].sum() / tot.volume_bits:.2f}")
        keep[a] = (model, la, pw)

        ax = axes[0, j]
        with torch.no_grad():
            p1 = model(G).softmax(-1)[:, 1].reshape(gx.shape).numpy()
        ax.contourf(gx, gy, p1, levels=20, cmap="RdBu_r", alpha=0.35, vmin=0, vmax=1)
        ax.contour(gx, gy, p1, levels=[0.5], colors="k", linewidths=0.8)
        sz = 4 + 60 * np.sqrt(pw["information_bits"] / 5)
        cols = np.where(y.numpy() == 1, "#B2182B", "#2166AC")
        Xn = X.numpy()
        ax.scatter(Xn[~flipped, 0], Xn[~flipped, 1], s=sz[~flipped], c=cols[~flipped], alpha=0.7, lw=0)
        ax.scatter(Xn[flipped, 0], Xn[flipped, 1], s=sz[flipped], c=cols[flipped], marker="X",
                   edgecolors="k", lw=0.5)
        ax.set_title(f"α = {a}   (P={tot.n_params}, γ={tot.dof:.0f} eff. params)\n"
                     f"code: model {tot.model_bits:.0f} + data {tot.data_bits:.0f} = "
                     f"{tot.model_bits + tot.data_bits:.0f} bits; flipped fit {mem:.0%}", fontsize=8.5)
        ax.set_xticks([]), ax.set_yticks([])
        if j == 0:
            ax.text(0.01, 0.99, "marker area ∝ capacity (bits)\n✕ = flipped label",
                    transform=ax.transAxes, va="top", fontsize=7)
        sc = capacity_scatter(axes[1, j], pw, flipped)
        if j == 0:
            scatter_legend(axes[1, j])
        if j == len(alphas) - 1:
            add_colorbar(fig, sc, axes[1, j])
    axes[0, 0].annotate("strong prior\n(more compression)", xy=(0.5, 1.22), xycoords="axes fraction",
                        ha="center", fontsize=8, color="0.3")
    axes[0, -1].annotate("weak prior\n(less compression)", xy=(0.5, 1.22), xycoords="axes fraction",
                         ha="center", fontsize=8, color="0.3")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig1_moons_capacity_vs_fidelity.png", dpi=160)
    plt.close(fig)
    return X, y, flipped, keep


# ============================================================================ #
# 2. Tiny ViT on 8x8 digits
# ============================================================================ #
class Block(nn.Module):
    def __init__(self, d, heads, d_mlp):
        super().__init__()
        self.h, self.dh = heads, d // heads
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.fc1, self.fc2 = nn.Linear(d, d_mlp), nn.Linear(d_mlp, d)

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).reshape(B, T, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        att = (q @ k.transpose(-1, -2) / math.sqrt(self.dh)).softmax(-1)
        x = x + self.proj((att @ v).transpose(1, 2).reshape(B, T, D))
        return x + self.fc2(F.gelu(self.fc1(self.ln2(x))))


class TinyViT(nn.Module):
    """8x8 image -> 16 patches of 2x2 + [CLS] -> 1 transformer block -> 10 logits."""

    def __init__(self, d=20, heads=2, d_mlp=40, depth=1, c=10):
        super().__init__()
        self.embed = nn.Linear(4, d)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.pos = nn.Parameter(0.02 * torch.randn(1, 17, d))
        self.blocks = nn.ModuleList([Block(d, heads, d_mlp) for _ in range(depth)])
        self.ln, self.head = nn.LayerNorm(d), nn.Linear(d, c)

    def forward(self, x):
        B = x.shape[0]
        p = x.reshape(B, 4, 2, 4, 2).permute(0, 1, 3, 2, 4).reshape(B, 16, 4)
        h = torch.cat([self.cls.expand(B, -1, -1), self.embed(p)], 1) + self.pos
        for blk in self.blocks:
            h = blk(h)
        return self.head(self.ln(h[:, 0]))


def demo_vit(alpha=8.0, n_train=None, flip=0.05, seed=0):
    n_train = n_train or CFG["vit_n"]
    print("\n== 2. Tiny ViT on sklearn digits ==")
    t = tic()
    d = load_digits()
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(d.target))
    tr, te = perm[:n_train], perm[n_train:]
    X = torch.tensor(d.data / 16.0, dtype=torch.float32)
    y_true = torch.tensor(d.target)
    Xtr, Xte, ytr_true, yte = X[tr], X[te], y_true[tr], y_true[te]
    flipped = rng.random(n_train) < flip
    fl = torch.from_numpy(flipped)
    ytr = ytr_true.clone()
    ytr[fl] = torch.tensor((ytr_true[fl].numpy() + rng.integers(1, 10, flipped.sum())) % 10)
    lik = Categorical()
    torch.manual_seed(seed)
    model = train_map(TinyViT(), Xtr, ytr, lik, alpha, adam_steps=CFG["vit_adam"], lr=3e-3, lbfgs_iters=CFG["vit_lbfgs"])
    with torch.no_grad():
        acc_te = (model(Xte).argmax(-1) == yte).float().mean().item()
        pred_tr = model(Xtr).argmax(-1)
    mem = (pred_tr[fl] == ytr[fl]).float().mean().item()
    t(f"trained: P={sum(p.numel() for p in model.parameters())}, test acc {acc_te:.3f}, "
      f"flipped labels fit {mem:.0%}")

    la = LinearizedLaplace(model, Xtr, ytr, lik, alpha)
    t("Jacobians + GGN + Cholesky + per-point covariances")
    pw = la.pointwise(n_samples=1000)
    tot = la.totals()
    t(f"γ={tot.dof:.0f} eff. params, entropy drop {tot.volume_bits:.0f} bits, model "
      f"{tot.model_bits:.0f} + data {tot.data_bits:.0f} bits; KL flipped/clean median "
      f"{np.median(pw['information_bits'][flipped]):.3f}/{np.median(pw['information_bits'][~flipped]):.4f}; "
      f"Σ unique entropy / total = {pw['volume_bits'].sum() / tot.volume_bits:.2f}")

    # set capacities: each class (as given) vs. sum of its members' unique capacities
    groups = []
    for c in range(10):
        idx = np.where((ytr.numpy() == c) & ~flipped)[0]
        groups.append((f"{c}", idx))
    groups.append(("flipped", np.where(flipped)[0]))
    groups.append(("random", rng.choice(np.where(~flipped)[0], flipped.sum(), replace=False)))
    gres = []
    for name, idx in groups:
        g = la.set_capacity(idx, n_samples=2000)
        gres.append((name, g, pw["volume_bits"][idx].sum(), pw["information_bits"][idx].sum()))
        print(f"    set {name:>7s} (n={g['n']:3d}): entropy {g['volume_bits']:7.1f} bits vs Σ single "
              f"{gres[-1][2]:6.1f};  KL {g['information_bits']:7.1f} vs Σ single {gres[-1][3]:6.1f}")
    t("set capacities")

    # ---- figure 2 -----------------------------------------------------------
    fig = plt.figure(figsize=(12, 8.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.9], hspace=0.55, wspace=0.5)
    ax = fig.add_subplot(gs[0, 0])
    sc = capacity_scatter(ax, pw, flipped, xlim=(1e-4, 40), s=9)
    scatter_legend(ax)
    add_colorbar(fig, sc, ax)
    ax.set_title(f"Tiny ViT (P={la.n_params}, α={alpha:g}) on digits, 5% flipped labels\n"
                 f"test acc {acc_te:.2f}; flipped labels fit {mem:.0%}", fontsize=9)

    ax = fig.add_subplot(gs[0, 1])
    ax.scatter(pw["volume_bits"][~flipped], pw["information_bits"][~flipped], s=8, c=CLEAN, alpha=0.5, lw=0)
    ax.scatter(pw["volume_bits"][flipped], pw["information_bits"][flipped], s=14, c=FLIP, marker="X", lw=0)
    lo, hi = 1e-4, 40
    ax.plot([lo, hi], [lo, hi], "k:", lw=0.7)
    ax.set_xscale("log"), ax.set_yscale("log"), ax.set_xlim(lo, hi), ax.set_ylim(lo, hi)
    ax.set_xlabel("entropy drop h(θ|D∖i) − h(θ|D)  [bits]\n(label-agnostic capacity allotted to xᵢ)")
    ax.set_ylabel("KL(q_D ‖ q_D∖i)  [bits]\n(information actually stored about (xᵢ,yᵢ))")
    ax.set_title("Volume vs. information", fontsize=9)

    ax = fig.add_subplot(gs[0, 2])
    names = [g[0] for g in gres]
    xs = np.arange(len(names))
    ratios = [g[1]["volume_bits"] / g[2] for g in gres]
    ax.bar(xs, ratios, color=["0.45"] * 10 + [FLIP, "0.75"])
    ax.axhline(1, color="k", lw=0.7, ls=":")
    ax.set_xticks(xs, names, rotation=45, fontsize=7)
    ax.set_ylim(0.95, max(ratios) * 1.04)
    ax.set_ylabel("joint ÷ Σ single-point entropy drop")
    ax.set_title("Set capacity: > 1 means the points\nshare (redundantly encode) information", fontsize=9)
    ax.set_xlabel("set: each digit class (clean), flipped labels, random clean points", fontsize=7.5)

    order = np.argsort(-pw["information_bits"])[:16]
    sub = gs[1, :].subgridspec(2, 8, wspace=0.1, hspace=0.45)
    probs = la.f.softmax(-1)
    for k, i in enumerate(order):
        ax = fig.add_subplot(sub[k // 8, k % 8])
        ax.imshow(Xtr[i].reshape(8, 8), cmap="gray_r")
        ax.set_xticks([]), ax.set_yticks([])
        col = FLIP if flipped[i] else "k"
        ax.set_title(f"y={ytr[i].item()} (true {ytr_true[i].item()})\np(y)={probs[i, ytr[i]]:.2f} "
                     f"{pw['information_bits'][i]:.1f}b", fontsize=7, color=col)
        for s in ax.spines.values():
            s.set_visible(True), s.set_color(col)
    fig.text(0.5, 0.455, "16 highest-capacity training images  (red = injected label flip; "
             "b = KL bits; p(y) = model's prob. of the given label)", ha="center", fontsize=9)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.94, bottom=0.02)
    fig.savefig(f"{OUT}/fig2_vit_digits.png", dpi=160)
    plt.close(fig)
    return Xtr, ytr, flipped, model, la, pw, alpha


# ============================================================================ #
# 3. Validation against actual retraining
# ============================================================================ #
def validate(X, y, model, la, pw, alpha, n_top, n_rand, seed=0):
    rng = np.random.default_rng(seed)
    top = np.argsort(-pw["information_bits"])[:n_top]
    rest = np.setdiff1d(np.arange(len(X)), top)
    idx = np.r_[top, rng.choice(rest, n_rand, replace=False)]
    lik = la.likelihood
    pred, act = [], []
    for i in idx:
        m2 = retrain_without(model, X, y, lik, alpha, [i], lbfgs_iters=CFG["retrain_iters"])
        with torch.no_grad():
            ll_new = lik.log_lik(m2(X[i:i + 1]).double(), y[i:i + 1]).item() / LN2
        act.append(pw["fidelity_bits"][i] - ll_new)
        pred.append(pw["fidelity_bits"][i] - pw["heldout_fidelity_bits"][i])
    return np.array(pred), np.array(act), n_top, idx


def fig_validation(results, pws):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.9))
    cols = ("#2a9d8f", "#e76f51")
    for k, ((name, (pred, act, n_top, idx)), c) in enumerate(zip(results.items(), cols)):
        ax = axes[k]
        ax.scatter(act[n_top:], pred[n_top:], s=18, c=c, alpha=0.8, label="random points")
        ax.scatter(act[:n_top], pred[:n_top], s=24, c=c, marker="^", alpha=0.8, label="highest capacity")
        hi = max(act.max(), pred.max()) * 1.1
        lo = min(act.min(), pred.min(), 0) - 0.05 * hi
        ax.plot([lo, hi], [lo, hi], "k:", lw=0.8)
        r = np.corrcoef(act, pred)[0, 1]
        sr = np.corrcoef(np.argsort(np.argsort(act)), np.argsort(np.argsort(pred)))[0, 1]
        ax.set_title(f"{name}\nPearson {r:.2f}, Spearman {sr:.2f}", fontsize=9)
        ax.set_xlabel("actual memorization [bits]\nlog₂p(yᵢ|θ*) − log₂p(yᵢ|θ*₋ᵢ), retrained")
        ax.set_ylabel("one-step (Newton) prediction [bits]")
        ax.legend(fontsize=7, frameon=False)
        axes[2].scatter(pws[name]["information_bits"][idx], act, s=16, c=c, alpha=0.8, label=name.split(":")[0])
    ax = axes[2]
    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=0.01)
    ax.set_xlabel("capacity: KL(q_D ‖ q_D∖i)  [bits]")
    ax.set_ylabel("actual memorization [bits]")
    ax.set_title("Does capacity predict what\nretraining without the point loses?", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig3_validation.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="figures", help="directory for figures")
    ap.add_argument("--quick", action="store_true", help="tiny settings for a smoke test")
    args = ap.parse_args()
    OUT = args.out
    os.makedirs(OUT, exist_ok=True)
    if args.quick:
        CFG.update(QUICK)
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    sanity_linear()
    Xm, ym, flm, keep = demo_moons()
    Xv, yv, flv, vit, la_v, pw_v, a_vit = demo_vit()

    print("\n== 3. Validation vs. retraining ==")
    t = tic()
    a_mid = 0.2
    model_m, la_m, pw_m = keep[a_mid]
    k_m, k_v = f"MLP moons: α={a_mid}", f"Tiny ViT digits: α={a_vit:g}"
    res = {k_m: validate(Xm, ym, model_m, la_m, pw_m, a_mid, *CFG["val_mlp"])}
    t("MLP retrains")
    res[k_v] = validate(Xv, yv, vit, la_v, pw_v, a_vit, *CFG["val_vit"])
    t("ViT retrains")
    fig_validation(res, {k_m: pw_m, k_v: pw_v})
    for k, (pred, act, n_top, idx) in res.items():
        kl = {k_m: pw_m, k_v: pw_v}[k]["information_bits"][idx]
        print(f"  {k}: Newton vs retrain Pearson r = {np.corrcoef(pred, act)[0, 1]:.3f}; "
              f"Spearman(KL, actual) = {np.corrcoef(np.argsort(np.argsort(kl)), np.argsort(np.argsort(act)))[0, 1]:.3f}")
