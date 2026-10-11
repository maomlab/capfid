# Case study: what do ADMET models memorize?

Dropout MLPs are trained on three [Therapeutics Data Commons](https://tdcommons.ai) ADMET
benchmark tasks with three molecular featurizations. `capfid` then measures how many bits of
each network's capacity every training compound costs. The compounds that cost the most are
depicted and placed in UMAP embeddings of each dataset.

| task | type | train / valid / test | source |
|---|---|---|---|
| `AMES` | mutagenicity, binary | 5,093 / 728 / 1,457 | several public Ames data sets |
| `Lipophilicity_AstraZeneca` | logD₇.₄, regression | 2,940 / 420 / 840 | one AstraZeneca laboratory |
| `Solubility_AqSolDB` | logS, regression | 6,986 / 999 / 1,995 | nine public sources |

Splits are the TDC benchmark scaffold splits (seed 1). Two AqSolDB test SMILES that RDKit
cannot parse are dropped.

**Features** (2,048 dimensions each):

- **ECFP4:** Morgan radius 2, folded to 2,048 bits.
- **APDP:** atom pairs plus Sheridan's donor/acceptor binding-property pairs, folded into one
  2,048-bit vector as in
  [maomlab/MPLearn](https://github.com/maomlab/MPLearn/blob/8d10bc41ea82934ebb45e00e7fd1d192cb135114/MPLearn/chemoinformatics/chemical_space.py#L38).
- **CheMeleon:** the [CheMeleon](https://github.com/JacksonBurns/chemeleon) foundation-model
  embedding.

**Models:** ReLU MLPs with dropout, chosen by a 24-configuration random search on the
validation split. Each configuration is analyzed in two training regimes:

- **early:** stopped at the validation-selected epoch.
- **long:** trained for 100 epochs.

**Capacity analysis:** `capfid.KernelLaplace` gives the exact linearized-Laplace posterior of
each 0.6–4.2M-parameter network. It works through the network's empirical neural tangent
kernel, so the cost depends on the number of training compounds, not the number of
parameters. Because these networks are not trained to a mode, the posterior mean is refit to
the MAP of the linearized network (`prior_mean=`). The prior precision α maximizes the
training-set (Gaussian-process) marginal likelihood. For regression, the noise σ maximizes
the predictive likelihood of the validation split, with α re-optimized at each σ:
maximizing the training marginal likelihood over σ as well lets a kernel that was trained on
the same labels explain their noise as signal, and gives overconfident predictions.

**Groups:** compounds that a featurization cannot tell apart (identical vectors), and
compounds that are the same constitution up to stereochemistry and protonation (shared
InChIKey skeleton), are scored jointly with `capfid.KernelLaplace.set_capacity`. Per-compound
scores understate such groups, because each member is redundant with the rest.

**Validation:** 48 networks per model are retrained on random 70% subsets. The memorization
of each compound measured this way (Feldman & Zhang 2020) is compared with capfid's
estimates.

## Running it

```bash
# Python >= 3.11, from the repository root
pip install -e ".[examples]" -r examples/tdc_casestudy/requirements.txt
pip install --no-deps PyTDC==0.4.1

python examples/tdc_casestudy/run_casestudy.py all --workers 10   # ~3 h on 10 CPU cores
python examples/tdc_casestudy/figures.py --out paper/figures
```

Each stage caches its outputs under `cache/` (gitignored). `figures.py` writes the
manuscript figures and these tables to `results/`:

| file | contents |
|---|---|
| `test_metrics.csv` | train, validation and test metrics for every model and seed |
| `capfid_summary.csv` | per-model posterior hyperparameters, totals and run time |
| `models.csv` | per task, featurization and regime: performance, Laplace totals, validation correlations |
| `per_compound_<task>.csv.gz` | every training compound: capfid scores for all models, memorization by retraining, predictions |
| `key_compounds_<task>.csv` | the compounds highlighted in the figures, with nearest training neighbors |
| `duplicate_groups.csv` | joint capacity (`set_capacity`) of every group of training compounds sharing a feature vector or a constitution |
| `collisions.csv` | those groups summarized per task and featurization, split by whether their labels conflict |

## What capfid says about these models

**It ranks what the networks actually memorize.** Per-compound information agrees with
memorization measured by 864 retrainings at Spearman 0.38–0.73, from one posterior per
network instead of 48 retrainings. Centering the posterior on the trained weights instead of
refitting (`prior_mean=`) is worse for 15 of the 18 models and produces single-compound
scores up to 226 bits, because dropout and early stopping leave ∇L(θ\*) ≠ 0.

**The totals are much smaller than the networks' own memorization** (4–400×) and compare
models only within a training regime. The posterior is calibrated to predict held-out
compounds, and it conditions on the trained network's tangent features, so label information
already written into those features is not counted.

**The compounds it flags are interpretable:**

- **AMES** — label conflicts. All 20 top-consensus compounds are mispredicted by all three
  networks, and 81% of the top 100 have a near neighbor (Tc ≥ 0.5) with the opposite label,
  against 15% of the training set. Most are negatives carrying classical mutagenicity alerts
  (aryl azides, nitroarenes, nitroso compounds, epoxides); the top compound is a
  diazido-acridine labeled negative whose regioisomer (Tc 0.91) is labeled positive.
- **Lipophilicity** — a missing mechanism. The top compounds are acids, bases, zwitterions
  and permanent cations, whose logD₇.₄ depends on pKa, which no representation encodes.
  Two labels look wrong (8-hydroxyquinoline-2-carboxylic acid at 3.30; a phosphate ester of
  an amino acid at 2.50).
- **AqSolDB** — scope and errors. Salts, metal compounds and other inorganics are 14% of the
  training set but 56–74% of each model's top 1%, including two freely soluble salts recorded
  near logS = −8.

**Representation collisions are a bigger problem than capacity.** APDP maps 1,886 AqSolDB
training compounds (27%) into groups with identical vectors, 1,002 of them in groups whose
labels span more than 1 log unit; the largest group is 149 compounds with an all-zero APDP
vector spanning 12 log units. Scored jointly with `set_capacity`, that group determines 0.72
effective parameters and stores 7.2 bits, yet costs 669 bits of data code: the model predicts
their mean and ignores them. ECFP4 collapses homologous series (16 n-alkanols spanning 6 log
units share one vector). CheMeleon separates nearly every compound.

**Memorization from longer training did not pay.** Training AMES networks to 100 epochs
instead of the validation-selected 1–5 multiplied memorization by retraining 4–13× and drove
train AUROC to 1.00, while test AUROC changed by −0.001 to −0.042. The per-compound ranking
is largely preserved between regimes (Spearman 0.56–0.88), so longer training mostly
memorizes compounds that already cost capacity.

**Prospective volume needs no label** and flags uncertain test compounds for AMES
(ρ = 0.57–0.67 with absolute test error, 3× error from the lowest to the highest quintile),
but barely for the regression tasks (ρ = 0.06–0.21), where most test error is label noise and
activity cliffs rather than distance from the training set.
