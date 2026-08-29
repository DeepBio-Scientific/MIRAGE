<div align="center">

# MIRAGE

**M**easuring **I**nterpolation and **R**edundancy in **A**ffinity **GE**neralization

*The headline r ≈ 0.6 is a mirage that dissolves on singleton families.*

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)
[![Dataset](https://img.shields.io/badge/🤗-dataset-yellow.svg)](https://huggingface.co/datasets/DeepBioScientific/mirage)

</div>

## Why

Frontier co-folding affinity models report Pearson ≈ 0.6, but that number conflates two
things: *interpolation within protein families already in public data* and *transfer to novel
targets*. MIRAGE measures the second. The central result is an **interaction** — co-folders
depend strongly on protein-family support; family-disjoint shallow controls do not.

Define the **family generalization gap** `G_m = r(support≥301) − r(support=1)` (matched-affinity,
two-level bootstrap):

| method | class | G_m | 95% CI |
|---|---|---|---|
| Nesso-1 | affinity co-folder | **+0.45** | [+0.27, +0.61] |
| Boltz-2 | affinity co-folder | **+0.62** | [+0.12, +1.01] |
| RF-QSAR | family-disjoint ML | −0.01 | [−0.16, +0.16] |
| ligand-kNN | family-disjoint ML | −0.03 | [−0.17, +0.11] |
| mol. weight | trivial | −0.01 | [−0.15, +0.15] |

Co-folder gaps are significantly positive; every family-disjoint / trivial control's spans zero.
The shallow controls aren't weaker co-folders — they behave *differently* as family redundancy
changes. Family support predicts accuracy after controlling for ligand similarity, protein length,
date, family affinity variance, measurement type, and affinity range (Nesso t=−8.1, Boltz t=−3.7),
and the dependence localizes to the **protein-family axis, not ligand chemistry**. A family-mean
predictor alone scores 0.564 — the memorization ceiling.

**Practical consequence:** on novel families a shallow family-disjoint random forest (0.41) beats
both billion-parameter co-folders (Nesso 0.32, Boltz 0.31). On a novel post-cutoff target
(OpenBind EV-A71 2A protease), neither co-folder reliably beats molecular weight (Nesso 0.49, MW
0.47, Boltz 0.40).

**It holds on the field's own benchmark, not just ours.** On the standard ATOM3D **LBA30**
(30%-identity split), where IPBind reports Pearson 0.732 and no paper has ever reported a
ligand-only control, we find **molecular weight alone scores 0.44** — matching or beating most of
the published leaderboard (ENN 0.39, DeepDTA 0.47, near 3DCNN 0.55 / GNN 0.55). The top methods
(IPBind, EHIGN, GIGN) do clear the ligand-only ceiling (0.46), so they learn real structure — but
the small margins reported by the rest were never checked against ligand size.

We call this **redundancy-driven inflation / training-familiarity dependence**, not "leakage" —
reserving that term for demonstrable train/test boundary crossing. MIRAGE measures the thing
that matters for a new drug program: **transfer to protein families you have not seen.**

📄 **Paper**: [`paper/mirage.pdf`](paper/) — full experiments, controls, and analysis.

## Models evaluated

MIRAGE audits methods across six classes. Any new model plugs in the same way.

| class | methods | on MIRAGE |
|---|---|---|
| **affinity co-folders** | Nesso-1, Boltz-2 | strong family-support dependence (G_m +0.45 / +0.62) |
| **confidence-proxy co-folders** (ipTM, not affinity) | Chai-1, ESMFold2 | Chai-1's ipTM *itself* carries the dependence (novel 0.16 → redundant 0.46, t=−3.8) with no affinity head; ESMFold2's ipTM is a weak, uncalibrated proxy (t=−0.7) |
| **docking scoring functions** | smina (physics), gnina (CNN) | the split is **learned-vs-physics**: smina family-flat (t=+4.5), gnina CNN family-dependent (t=−4.5) like the co-folders |
| **family-disjoint ML controls** | RF-QSAR, ligand-kNN | flat across family support (G_m ≈ 0); **beat the co-folders on novel families** |
| **trivial / identity baselines** | molecular weight, clogp, family-mean | family-mean alone = 0.564 (memorization ceiling); MW ties the co-folders on the novel target |
| **temporal-arm baselines** (OpenBind) | gnina, smina, AEV-PLIG, AQ-Affinity | none reliably beat molecular weight |
| **pose prediction** | smina (physics), SigmaDock + DiffDock-L (DL diffusion), Chai-1 + Boltz-2 (co-folders, ±MSA) | physics difficulty floor +0.23; SigmaDock +0.28 (generalizes), DiffDock-L +0.36 (**collapses on novel: 18%**); single-sequence co-folders are worst (**Boltz-2 +0.64, novel 2%**), but an **MSA rescues novel families** (+0.64→+0.20; novel 2%→46%, 22/50 fail→OK flips, 0 reverse) while leaving seen families untouched → the deficit is *missing evolutionary signal*, not difficulty. DiffDock-L adapted to run on Blackwell (torch 2.7/cu128) |

That Chai-1 (structure confidence, *no* affinity head) reproduces the family-support dependence
shows it lives in the learned structural representation — not just a trained affinity head.

> **AlphaFold3** is a pluggable stub ([`mirage/cofolders.py`](mirage/cofolders.py)): its code
> is open but weights are request-only, and it has no affinity head. Provide weights and it runs
> as an ipTM proxy with one call.

## Install

```bash
pip install mirage                 # core
pip install "mirage[baselines]"    # + rdkit, to run the MW/clogp baselines
pip install "mirage[hub]"          # + datasets, to pull data from the Hub
```

Or from source:

```bash
git clone https://github.com/DeepBioScientific/mirage && cd mirage
pip install -e ".[baselines,dev]"
```

## Use it

Wrap your model as `predict(sequence, smiles) -> float` (higher = stronger binder):

```python
import mirage

def my_model(sequence, smiles):
    return my_net.predict_affinity(sequence, smiles)

red = mirage.run_redundancy(my_model)   # balanced 3,360-target quick set
tmp = mirage.run_temporal(my_model)     # novel target vs molecular weight
print(red.summary())
print(tmp.summary())
```

If your score is "lower = stronger" (e.g. `log10(IC50)`), pass `higher_is_stronger=False`.

**No Python?** Score a CSV instead:

```bash
mirage score redundancy preds.csv          # columns: id, prediction
mirage score temporal   preds.csv --lower-is-stronger
```

See [`examples/`](examples/) for a runnable template and the CSV workflow.

## What you get

- **Family-size dose-response** — per-bin correlation, matched-pK so label spread cannot
  confound it, with bootstrap CIs.
- **Family generalization gap `G_m`** — `r(support≥301) − r(support=1)`, with a two-level
  (family→ligand) bootstrap CI. The headline: large and positive means accuracy is bought by
  family support.
- **Novel-family Pearson** — the single number to report: accuracy on singleton + small
  families.
- **Family-support slope** — regression of absolute error on `log10(family_size)`, controlling
  for the affinity regime and covariates; a negative slope means accuracy tracks how familiar
  the family is.
- **Temporal arm** — within-target Spearman on a novel target, head-to-head against
  molecular weight and clogp.

## Datasets

Two configs, shipped as parquet and on the [Hub](https://huggingface.co/datasets/DeepBioScientific/mirage):

- **`redundancy`** — 18,759 PDBbind-derived complexes with `family_size` annotations
  (`in_core` marks a balanced 3,360 quick-eval subset).
- **`temporal`** — 649 compounds against one novel post-2021 target (OpenBind, CC0), with
  molecular-weight / clogp / Boltz-2 baselines.

```python
df = mirage.load("redundancy", core=True)
df = mirage.load("temporal")
```

Regenerate from source with [`scripts/build_dataset.py`](scripts/build_dataset.py);
push your own copy to the Hub with [`scripts/push_to_hub.py`](scripts/push_to_hub.py).

## Reporting results

When you report a MIRAGE number, please quote the **family generalization gap `G_m`** and the
**novel-family Pearson**, not just the overall correlation — that is the whole point. A model that
scores 0.6 overall but 0.1 on novel families should say so. Prefer equal-family weighting and the
two-level bootstrap; report the OpenBind score as *one external target*, not population-level proof.

## Provenance & licensing

Code is MIT. The `redundancy` set is derived from **PDBbind** (only derived annotations are
redistributed; observe PDBbind's terms for the underlying structures). The `temporal` set is
derived from the **OpenBind A71EV2A** release (CC0). Family sizes use MMseqs2 at 30% identity.

## Citation

```bibtex
@software{mirage2026,
  title  = {MIRAGE: Measuring Interpolation and Redundancy in Affinity GEneralization},
  year   = {2026},
  url    = {https://github.com/DeepBioScientific/mirage}
}
```
