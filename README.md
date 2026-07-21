<div align="center">

# FamBench

**Family-Stratified Protein–Ligand Affinity Benchmark**

*Does your affinity model generalise to novel targets, or is its accuracy carried by
protein families it has seen many times?*

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)
[![Dataset](https://img.shields.io/badge/🤗-dataset-yellow.svg)](https://huggingface.co/datasets/mrnafold/fambench)

</div>

## Why

Frontier co-folding affinity models report Pearson ≈ 0.6 on standard benchmarks. But those
test sets are dominated by protein families heavily represented in the PDB. Stratify accuracy
by **protein-family redundancy** (matched-pK, label spread held constant) and the picture
changes sharply — and the key controls tell you *why*:

| protein family size | Nesso-1 | Boltz-2 | RF-QSAR* | ligand-kNN* |
|---|---|---|---|---|
| 1 (singleton, novel) | +0.11 | **−0.03** | +0.26 | +0.22 |
| 2–5 | +0.23 | −0.01 | +0.26 | +0.22 |
| 6–20 | +0.38 | −0.01 | +0.26 | +0.19 |
| 21–80 | +0.42 | +0.42 | +0.25 | +0.24 |
| 81–300 | +0.44 | +0.37 | +0.20 | +0.17 |
| 301+ (redundant) | +0.55 | +0.59 | +0.18 | +0.18 |

*\*RF-QSAR and ligand-kNN are evaluated under **family-disjoint** cross-validation — they
cannot memorize the test family, and they are **flat**. The co-folders rise steeply. That gap
is the leakage.* On **novel families a shallow family-disjoint random forest (0.41) beats both
billion-parameter co-folders** (Nesso 0.32, Boltz 0.31).

On a genuinely novel, post-cutoff target (OpenBind EV-A71 2A protease), **neither co-folder
reliably beats a molecular-weight baseline** (Nesso 0.49, MW 0.47, Boltz 0.40 Spearman).

A single headline correlation hides all of this. FamBench measures the thing that matters for
a new drug program: **accuracy on protein families you have not seen.**

📄 **Paper**: [`paper/fambench.pdf`](paper/) — full experiments, controls, and analysis.

## Install

```bash
pip install fambench                 # core
pip install "fambench[baselines]"    # + rdkit, to run the MW/clogp baselines
pip install "fambench[hub]"          # + datasets, to pull data from the Hub
```

Or from source:

```bash
git clone https://github.com/mrnafold/fambench && cd fambench
pip install -e ".[baselines,dev]"
```

## Use it

Wrap your model as `predict(sequence, smiles) -> float` (higher = stronger binder):

```python
import fambench

def my_model(sequence, smiles):
    return my_net.predict_affinity(sequence, smiles)

red = fambench.run_redundancy(my_model)   # balanced 3,360-target quick set
tmp = fambench.run_temporal(my_model)     # novel target vs molecular weight
print(red.summary())
print(tmp.summary())
```

If your score is "lower = stronger" (e.g. `log10(IC50)`), pass `higher_is_stronger=False`.

**No Python?** Score a CSV instead:

```bash
fambench score redundancy preds.csv          # columns: id, prediction
fambench score temporal   preds.csv --lower-is-stronger
```

See [`examples/`](examples/) for a runnable template and the CSV workflow.

## What you get

- **Family-size dose-response** — per-bin correlation, matched-pK so label spread cannot
  confound it, with bootstrap CIs.
- **Novel-family Pearson** — the single number to report: accuracy on singleton + small
  families.
- **Leakage slope** — regression of absolute error on `log10(family_size)`, controlling for
  the affinity regime; a negative slope means accuracy is bought by redundancy.
- **Temporal arm** — within-target Spearman on a novel target, head-to-head against
  molecular weight and clogp.

## Datasets

Two configs, shipped as parquet and on the [Hub](https://huggingface.co/datasets/mrnafold/fambench):

- **`redundancy`** — 18,759 PDBbind-derived complexes with `family_size` annotations
  (`in_core` marks a balanced 3,360 quick-eval subset).
- **`temporal`** — 649 compounds against one novel post-2021 target (OpenBind, CC0), with
  molecular-weight / clogp / Boltz-2 baselines.

```python
df = fambench.load("redundancy", core=True)
df = fambench.load("temporal")
```

Regenerate from source with [`scripts/build_dataset.py`](scripts/build_dataset.py);
push your own copy to the Hub with [`scripts/push_to_hub.py`](scripts/push_to_hub.py).

## Reporting results

When you report a FamBench number, please quote the **novel-family Pearson** and the
**leakage slope**, not just the overall correlation — that is the whole point. A model that
scores 0.6 overall but 0.1 on novel families should say so.

## Provenance & licensing

Code is MIT. The `redundancy` set is derived from **PDBbind** (only derived annotations are
redistributed; observe PDBbind's terms for the underlying structures). The `temporal` set is
derived from the **OpenBind A71EV2A** release (CC0). Family sizes use MMseqs2 at 30% identity.

## Citation

```bibtex
@software{fambench2026,
  title  = {FamBench: Family-Stratified Protein-Ligand Affinity Benchmark},
  year   = {2026},
  url    = {https://github.com/mrnafold/fambench}
}
```
