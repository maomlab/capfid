"""Build a self-contained interactive dashboard for the TDC case study.

Bakes the cached per-compound capfid tables, UMAP embeddings and collision groups into one
static HTML file that needs no server and no Python:

    python examples/tdc_casestudy/dashboard.py            # -> examples/tdc_casestudy/dashboard.html

Run after ``run_casestudy.py all`` and ``figures.py`` (which write the caches this reads).
Molecules are depicted in the browser by RDKit.js, loaded from a CDN on first use; with no
network the dashboard still works and shows SMILES instead of structures.
"""
from __future__ import annotations

import argparse
import base64
import gzip
import json
import pathlib

import numpy as np
import pandas as pd
from scipy.special import expit

from common import FEATURES, RESULTS, TASKS, load_features, load_task
from figures import REGIMES, capfid_table, key_compounds, retraining_memorization, summary
from figures import umap_embedding

HERE = pathlib.Path(__file__).resolve().parent

# Per-compound quantities carried into the dashboard. "train" columns come from the
# training-set capfid table, "test" columns from the test-set one, where only the quantities
# that need no label are defined: the prospective volume and the predictions. Held-out
# fidelity is left out because it is exactly fidelity - memorization.
TRAIN_COLS = {
    "dof": "dof",
    "volume": "volume_bits",
    "information": "information_bits",
    "memorization": "memorization_bits",
    "fidelity": "fidelity_bits",
}
TEST_COLS = {"volume": "volume_bits", "fidelity": "fidelity_bits"}
#: which splits each per-compound quantity is defined on. It travels in the payload, and the
#: dashboard greys out and excludes the entries outside it rather than plotting them as zero.
DOMAIN = {"dof": "train", "information": "train", "memorization": "train", "retrain": "train",
          "volume": "train+test", "fidelity": "train+test", "pred": "train+test",
          "gp": "train+test", "gp_heldout": "train"}


def enc(x, tiny=1e-7, sig=3):
    """Round an array to ``sig`` significant digits for JSON; NaN and |x| < tiny become 0.

    Rounding is what keeps the embedded payload small, and these are display quantities: a
    score of 1e-9 bits and one of 0 bits mean the same thing. Undefined entries also become 0
    rather than null; the dashboard masks them by split using ``DOMAIN``.
    """
    v = np.asarray(x, dtype=float)
    v = np.where(np.isfinite(v) & (np.abs(v) >= tiny), v, 0.0)
    return [float(f"{z:.{sig}g}") for z in v]


def neighbors(task, k=3, block=512):
    """For every compound, its ``k`` nearest ECFP4 Tanimoto neighbors among the training set
    (excluding itself). Near neighbors with opposite labels are the main thing one wants to
    see next to a compound that costs a model a lot of capacity."""
    df = load_task(task)
    X = load_features(task, "ECFP4").astype(np.float32)
    tr = np.flatnonzero((df.split == "train").to_numpy())
    P = X[tr]
    pn = P.sum(1)
    idx, sim = [], []
    for i in range(0, len(X), block):
        Q = X[i:i + block]
        inter = Q @ P.T
        s = inter / (Q.sum(1)[:, None] + pn[None, :] - inter + 1e-9)
        for j, q in enumerate(range(i, i + len(Q))):
            row = s[j]
            self_at = np.flatnonzero(tr == q)
            if len(self_at):
                row = row.copy()
                row[self_at[0]] = -1.0
            top = np.argpartition(-row, k)[:k]
            top = top[np.argsort(-row[top])]
            idx.append([int(tr[t]) for t in top])
            sim.append([round(float(row[t]), 2) for t in top])
    return idx, sim


def task_payload(task, seed=0):
    df = load_task(task)
    short = TASKS[task]["short"]
    n = len(df)
    split_code = df.split.map({"train": 0, "valid": 1, "test": 2}).to_numpy()
    S = summary()

    models = {}
    for f in FEATURES:
        for r in REGIMES:
            tr = capfid_table(task, f, r, seed=seed)
            te = capfid_table(task, f, r, seed=seed, split="test")
            s0 = S[(S.task == task) & (S.feature == f) & (S.regime == r)
                   & (S.seed == seed)].iloc[0]
            mem = retraining_memorization(task, f, r, s0.sigma)
            cols = {k: np.full(n, np.nan) for k in DOMAIN}
            ti, ei = tr.row.to_numpy(), te.row.to_numpy()
            for key, src in TRAIN_COLS.items():
                cols[key][ti] = tr[src].to_numpy()
            for key, src in TEST_COLS.items():
                cols[key][ei] = te[src].to_numpy()
            cols["retrain"][ti] = mem.retrain_memorization_bits.to_numpy()
            # predictions back in the task's own units: probability, logD or logS
            if TASKS[task]["kind"] == "binary":
                to = expit
            else:
                y = df.Y.to_numpy(float)[split_code == 0]
                mu, sd = y.mean(), y.std()
                to = lambda v: np.asarray(v) * sd + mu               # noqa: E731
            preds = (("pred", "network_output"), ("gp", "output"), ("gp_heldout", "heldout_output"))
            for key, src in preds:
                cols[key][ti] = to(tr[src].to_numpy())
                if src in te:
                    cols[key][ei] = to(te[src].to_numpy())
            pred_keys = {k for k, _ in preds}
            models[f"{f}|{r}"] = {
                k: enc(v, **({"tiny": 1e-6, "sig": 4} if k in pred_keys else {}))
                for k, v in cols.items()}

    G = pd.read_csv(RESULTS / "duplicate_groups.csv")
    G = G[G.task == task]
    groups = [dict(feature=g.feature, regime=g.regime, kind=g.kind,
                   members=[int(m) for m in str(g.members).split(";")], n=int(g.n),
                   y_min=round(float(g.y_min), 4), y_max=round(float(g.y_max), 4),
                   dof=round(float(g.dof), 4), volume=round(float(g.volume_bits), 4),
                   information=round(float(g.information_bits), 4),
                   singles=round(float(g.sum_single_information_bits), 4),
                   redundancy=round(float(g.redundancy), 4),
                   code_bits=round(float(-g.fidelity_bits), 4))
              for g in G.itertuples()]

    kp = RESULTS / f"key_compounds_{short}.csv"
    K = pd.read_csv(kp) if kp.exists() else key_compounds(task)
    keys = [dict(key=int(k.key), reason=k.reason, row=int(k.row)) for k in K.itertuples()]

    nn_idx, nn_sim = neighbors(task)
    return dict(
        name=task, short=short, kind=TASKS[task]["kind"], metric=TASKS[task]["metric"],
        unit=TASKS[task]["unit"], n=n,
        ids=df.Drug_ID.astype(str).tolist(), smiles=df.Drug.tolist(),
        y=enc(df.Y.to_numpy(float), tiny=0, sig=6), split=split_code.tolist(),
        umap={f: dict(zip("xy", (enc(c, tiny=0, sig=4) for c in umap_embedding(task, f).T)))
              for f in FEATURES},
        models=models, groups=groups, keys=keys, nn_idx=nn_idx, nn_sim=nn_sim)


def payload(seed=0):
    def records(path):
        return json.loads(pd.read_csv(path).to_json(orient="records"))

    return dict(
        features=list(FEATURES), regimes=list(REGIMES), seed=seed, domain=DOMAIN,
        built=pd.Timestamp.today().strftime("%Y-%m-%d"),
        tasks={TASKS[t]["short"]: task_payload(t, seed) for t in TASKS},
        model_table=records(RESULTS / "models.csv"),
        collisions=records(RESULTS / "collisions.csv"))


def build(out, seed=0):
    data = payload(seed)
    raw = json.dumps(data, separators=(",", ":"), allow_nan=False).encode()
    blob = base64.b64encode(gzip.compress(raw, 9)).decode()
    template = (HERE / "dashboard_template.html").read_text()
    html = template.replace("/*__PAYLOAD__*/", blob).replace("__BUILT__", data["built"])
    out = pathlib.Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    print(f"{out}: {len(html) / 1e6:.2f} MB "
          f"(payload {len(raw) / 1e6:.2f} MB json -> {len(blob) / 1e6:.2f} MB gzip+base64)")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(HERE / "dashboard.html"))
    ap.add_argument("--seed", type=int, default=0, help="which trained network to analyze")
    args = ap.parse_args()
    build(args.out, args.seed)
