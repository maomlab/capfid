"""capfid quickstart: which training examples does a small network spend its capacity on?

Two moons with 10% flipped labels, a tanh MLP trained to a MAP estimate, then per-example
capacity (bits stored in the weights) versus fidelity (log-likelihood). Runs in ~10 s on a CPU.

    pip install "capfid[examples]"   # or: pip install -e ".[examples]"
    python examples/quickstart.py
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import make_moons

import capfid
from capfid.plotting import capacity_fidelity_plot

# --- data with injected label noise -------------------------------------------------------
X, y = make_moons(300, noise=0.2, random_state=0)
flipped = np.random.default_rng(0).random(len(y)) < 0.10
y[flipped] = 1 - y[flipped]
X, y = torch.tensor(X, dtype=torch.float32), torch.tensor(y)

# --- train to a MAP estimate of  Σ_i ℓ_i + (α/2)‖θ‖²  -------------------------------------
alpha = 0.3                                   # prior precision = L2 coefficient on the summed loss
torch.manual_seed(1)
model = nn.Sequential(nn.Linear(2, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh(), nn.Linear(32, 2))
lik = capfid.Categorical()
capfid.train_map(model, X, y, lik, alpha, adam_steps=3000, lr=1e-2, lbfgs_iters=1000)

# --- capacity and fidelity ----------------------------------------------------------------
la = capfid.LinearizedLaplace(model, X, y, lik, prior_precision=alpha)
res = la.pointwise()
tot = la.totals()

print(la)
print(f"effective parameters γ = {tot.dof:.1f} of {tot.n_params}")
print(f"code length: model {tot.model_bits:.0f} + data {tot.data_bits:.0f} "
      f"= {tot.description_length_bits:.0f} bits")
print(f"single examples account for {tot.unique_fraction:.0%} of the model's information\n")

order = np.argsort(-res["information_bits"])[:10]
print(" idx  flipped  capacity(bits)  fidelity(bits)  held-out(bits)")
for i in order:
    print(f"{i:4d}  {str(bool(flipped[i])):7s}  {res['information_bits'][i]:14.3f}  "
          f"{res['fidelity_bits'][i]:14.3f}  {res['heldout_fidelity_bits'][i]:14.3f}")

med = lambda m: np.median(res["information_bits"][m])  # noqa: E731
print(f"\nmedian capacity: flipped {med(flipped):.3f} bits, clean {med(~flipped):.4f} bits")

s = la.set_capacity(np.where(flipped)[0])
print(f"all flipped labels jointly: volume {s['volume_bits']:.1f} bits "
      f"(redundancy {s['redundancy']:.2f})")

# --- plot ---------------------------------------------------------------------------------
import matplotlib  # noqa: E402

matplotlib.use("Agg")
ax = capacity_fidelity_plot(res, highlight=flipped, highlight_label="flipped label")
ax.set_title(f"two moons, α = {alpha}")
ax.figure.tight_layout()
ax.figure.savefig("quickstart_capacity_fidelity.png", dpi=150)
print("wrote quickstart_capacity_fidelity.png")
