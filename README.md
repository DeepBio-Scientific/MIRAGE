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
test sets are dominated by protein families that are heavily represented in the PDB. When
you stratify accuracy by **protein-family redundancy**, the picture changes sharply:

| protein family size | Boltz-2 | Nesso-1 |
|---|---|---|
| 1 (singleton, novel) | **−0.03** | +0.32 |
| 2–5 | −0.01 | +0.54 |
| 6–20 | −0.01 | +0.41 |
| 21–80 | +0.42 | +0.67 |
| 81–300 | +0.37 | +0.62 |
| 301+ (redundant) | +0.59 | +0.52 |

*Per-bin Pearson vs experimental pK, label spread held constant. On novel families both
models drop toward zero.* And on a genuinely novel, post-cutoff target (OpenBind EV-A71 2A
protease), **neither model reliably beats a molecular-weight baseline**:

| method | Spearman ρ |
|---|---|
| molecular weight | 0.47 |
| Nesso-1 | 0.49 |
| Boltz-2 | 0.40 |

A single headline correlation hides this. FamBench measures the thing that matters for a
new drug program: **accuracy on protein families you have not seen.**

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
