"""Shared data, featurization, model, and training code for the TDC case study.

Three Therapeutics Data Commons ADMET benchmark tasks (scaffold split, seed 1), three
featurizations (ECFP4, APDP, CheMeleon), and a dropout MLP tuned on the validation split.
Everything is cached under ``cache/`` next to this file.
"""
from __future__ import annotations

import json
import math
import pathlib
import warnings
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = pathlib.Path(__file__).resolve().parent
CACHE = HERE / "cache"
RESULTS = HERE / "results"
LN2 = math.log(2.0)

TASKS = {
    "AMES": dict(kind="binary", metric="AUROC", short="AMES", unit=""),
    "Lipophilicity_AstraZeneca": dict(kind="regression", metric="MAE", short="Lipo", unit="logD"),
    "Solubility_AqSolDB": dict(kind="regression", metric="MAE", short="AqSol", unit="logS"),
}
FEATURES = ("ECFP4", "APDP", "CheMeleon")
N_BITS = 2048


# ------------------------------------------------------------------------------- data
def load_task(name: str) -> pd.DataFrame:
    """TDC ADMET benchmark group task with the default (scaffold) train/valid split, seed 1.

    Returns one frame with columns Drug_ID, Drug (SMILES), Y, split in {train, valid, test}.
    Rows whose SMILES RDKit cannot parse are dropped (two AqSolDB test compounds).
    """
    path = CACHE / "splits" / f"{name}.csv"
    if path.exists():
        return pd.read_csv(path)
    from rdkit import Chem, RDLogger
    from tdc.benchmark_group import admet_group

    RDLogger.DisableLog("rdApp.*")

    group = admet_group(path=str(CACHE / "tdc"))
    bench = group.get(name)
    train, valid = group.get_train_valid_split(benchmark=bench["name"], split_type="default", seed=1)
    df = pd.concat([train.assign(split="train"), valid.assign(split="valid"),
                    bench["test"].assign(split="test")], ignore_index=True)
    ok = np.array([Chem.MolFromSmiles(smi) is not None for smi in df.Drug])
    if not ok.all():
        print(f"{name}: dropping {(~ok).sum()} unparseable SMILES: {df.Drug[~ok].tolist()}")
    df = df[ok].reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


# -------------------------------------------------------------------------- features
def _mols(smiles):
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    bad = [s for s, m in zip(smiles, mols) if m is None]
    if bad:
        raise ValueError(f"RDKit could not parse {len(bad)} SMILES, e.g. {bad[:3]}")
    return mols


def ecfp4(smiles, n_bits=N_BITS):
    """Morgan fingerprint, radius 2, folded to ``n_bits`` bits (ECFP4)."""
    from rdkit.Chem import rdFingerprintGenerator

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=n_bits)
    return np.stack([gen.GetFingerprintAsNumPy(m) for m in _mols(smiles)]).astype(np.float32)


def apdp(smiles, n_bits=N_BITS):
    """APDP: atom pairs (Carhart et al. 1985) plus Sheridan's donor/acceptor binding-property
    pairs, folded together into ``n_bits`` bits as in maomlab/MPLearn
    (``chemoinformatics/chemical_space.py``; Sheridan 2019)."""
    from rdkit.Chem.AtomPairs import Pairs, Sheridan

    out = np.zeros((len(smiles), n_bits), np.float32)
    for k, mol in enumerate(_mols(smiles)):
        for i in Pairs.GetAtomPairFingerprint(mol).GetNonzeroElements():
            out[k, i % n_bits] = 1
        for i in Sheridan.GetBPFingerprint(mol).GetNonzeroElements():
            out[k, (i + 8388608) % n_bits] = 1
    return out


def chemeleon(smiles, batch_size=256):
    """CheMeleon learned fingerprint (Burns et al. 2025): mean-aggregated atom embeddings of
    the message-passing encoder pretrained to predict Mordred descriptors. Adapted from
    ``chemeleon_fingerprint.py`` in github.com/JacksonBurns/chemeleon (MIT license)."""
    from pathlib import Path
    from urllib.request import urlretrieve

    from chemprop import featurizers, nn as cnn
    from chemprop.data import BatchMolGraph
    from chemprop.models import MPNN

    ckpt = Path.home() / ".chemprop" / "chemeleon_mp.pt"
    if not ckpt.exists():
        ckpt.parent.mkdir(exist_ok=True)
        urlretrieve("https://zenodo.org/records/15460715/files/chemeleon_mp.pt", ckpt)
    state = torch.load(ckpt, weights_only=True)
    mp = cnn.BondMessagePassing(**state["hyper_parameters"])
    mp.load_state_dict(state["state_dict"])
    model = MPNN(message_passing=mp, agg=cnn.MeanAggregation(),
                 predictor=cnn.RegressionFFN(input_dim=mp.output_dim)).eval()
    feat = featurizers.SimpleMoleculeMolGraphFeaturizer()
    mols = _mols(smiles)
    out = []
    with torch.no_grad():
        for i in range(0, len(mols), batch_size):
            bmg = BatchMolGraph([feat(m) for m in mols[i:i + batch_size]])
            out.append(model.fingerprint(bmg).numpy())
    return np.concatenate(out).astype(np.float32)


FEATURIZERS = {"ECFP4": ecfp4, "APDP": apdp, "CheMeleon": chemeleon}


def load_features(task: str, feature: str) -> np.ndarray:
    """Feature matrix for all rows of ``load_task(task)`` (cached)."""
    path = CACHE / "features" / f"{task}_{feature}.npy"
    if path.exists():
        return np.load(path)
    df = load_task(task)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X = FEATURIZERS[feature](df.Drug.tolist())
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, X)
    return X


@dataclass
class Data:
    """Model-ready tensors for one task and featurization (targets standardized on train)."""
    task: str
    feature: str
    kind: str
    df: pd.DataFrame
    X: dict
    y: dict
    y_mean: float
    y_std: float

    def y_raw(self, split):
        return self.y[split].numpy().ravel() * self.y_std + self.y_mean


def get_data(task: str, feature: str) -> Data:
    df = load_task(task)
    X = load_features(task, feature)
    kind = TASKS[task]["kind"]
    tr = (df.split == "train").to_numpy()
    if feature == "CheMeleon":                         # continuous: standardize on train
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        X = (X - mu) / sd
    y = df.Y.to_numpy(np.float64)
    y_mean, y_std = (0.0, 1.0) if kind == "binary" else (y[tr].mean(), y[tr].std())
    ys = ((y - y_mean) / y_std).astype(np.float32)
    Xs, Ys = {}, {}
    for s in ("train", "valid", "test"):
        m = (df.split == s).to_numpy()
        Xs[s] = torch.tensor(X[m], dtype=torch.float32)
        Ys[s] = torch.tensor(ys[m]).unsqueeze(1)
    return Data(task, feature, kind, df, Xs, Ys, float(y_mean), float(y_std))


# ---------------------------------------------------------------------- model, training
class MLP(nn.Module):
    """ReLU MLP with dropout after every hidden layer and one output (logit or mean)."""

    def __init__(self, d_in, width, depth, dropout):
        super().__init__()
        layers, d = [], d_in
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.ReLU(), nn.Dropout(dropout)]
            d = width
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


@dataclass
class HParams:
    width: int
    depth: int
    dropout: float
    weight_decay: float
    lr: float
    batch_size: int = 128
    epochs: int = 200

    def as_dict(self):
        return asdict(self)


def loss_fn(kind, f, y):
    """Mean training loss: binary cross-entropy, or ½·MSE on standardized targets."""
    if kind == "binary":
        return F.binary_cross_entropy_with_logits(f, y)
    return 0.5 * F.mse_loss(f, y)


def metric(kind, y_true, pred):
    """TDC leaderboard metric: AUROC (binary, higher is better) or MAE (regression)."""
    from sklearn.metrics import roc_auc_score

    if kind == "binary":
        return float(roc_auc_score(y_true, pred))
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(pred))))


def better(kind, a, b):
    return a > b if kind == "binary" else a < b


def predict(model, X, batch_size=4096):
    model.eval()
    with torch.no_grad():
        return torch.cat([model(X[i:i + batch_size]) for i in range(0, len(X), batch_size)])


def evaluate(model, data: Data, split: str) -> float:
    f = predict(model, data.X[split]).numpy().ravel()
    if data.kind == "binary":
        return metric("binary", data.y[split].numpy().ravel(), f)
    return metric("regression", data.y_raw(split), f * data.y_std + data.y_mean)


def train_mlp(data: Data, hp: HParams, seed: int, keep=None, early_stopping=False, patience=25):
    """Minibatch Adam with coupled weight decay (i.e. L2 on the mean loss) and dropout.

    ``keep`` optionally restricts the training set (boolean mask). With ``early_stopping``,
    the validation metric is checked every epoch and the best epoch is restored.
    Returns (model, best_epoch, best_valid_metric).
    """
    torch.manual_seed(seed)
    X, y = data.X["train"], data.y["train"]
    if keep is not None:
        X, y = X[torch.as_tensor(keep)], y[torch.as_tensor(keep)]
    model = MLP(X.shape[1], hp.width, hp.depth, hp.dropout)
    opt = torch.optim.Adam(model.parameters(), lr=hp.lr, weight_decay=hp.weight_decay)
    g = torch.Generator().manual_seed(seed)
    best, best_epoch, best_state, bad = None, hp.epochs, None, 0
    for epoch in range(1, hp.epochs + 1):
        model.train()
        perm = torch.randperm(len(X), generator=g)
        for i in range(0, len(X), hp.batch_size):
            b = perm[i:i + hp.batch_size]
            opt.zero_grad()
            loss_fn(data.kind, model(X[b]), y[b]).backward()
            opt.step()
        if early_stopping:
            m = evaluate(model, data, "valid")
            if best is None or better(data.kind, m, best):
                best, best_epoch, bad = m, epoch, 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, best_epoch, best


def search_space(n_configs=24, seed=0):
    """Random search over width, depth, dropout, weight decay and learning rate."""
    rng = np.random.default_rng(seed)
    grid = dict(width=[256, 512, 1024], depth=[1, 2, 3], dropout=[0.1, 0.2, 0.3, 0.5],
                weight_decay=[1e-6, 1e-5, 1e-4, 1e-3], lr=[1e-4, 3e-4, 1e-3])
    seen, out = set(), []
    while len(out) < n_configs:
        c = tuple(rng.choice(len(v)) for v in grid.values())
        if c in seen:
            continue
        seen.add(c)
        out.append(HParams(**{k: type(v[0])(v[j]) for (k, v), j in zip(grid.items(), c)}))
    return out


def save_json(obj, path):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=float))


def load_json(path):
    return json.loads(pathlib.Path(path).read_text())
