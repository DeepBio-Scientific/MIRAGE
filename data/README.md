---
license: mit
task_categories:
  - tabular-regression
tags:
  - binding-affinity
  - drug-discovery
  - protein-ligand
  - benchmark
  - data-leakage
pretty_name: FamBench
configs:
  - config_name: redundancy
    data_files: redundancy/test-*.parquet
    default: true
  - config_name: temporal
    data_files: temporal/test-*.parquet
---

# FamBench: Family-Stratified Protein–Ligand Affinity Benchmark

A benchmark for measuring whether a protein–ligand **binding-affinity** model generalises
beyond protein families that are heavily represented in the PDB, or whether its reported
accuracy is carried by that redundancy.

Frontier co-folding affinity models (Boltz-2, Nesso-1, …) report Pearson ≈ 0.6 on standard
benchmarks. When the same models are stratified by protein-family size, that accuracy is
concentrated on well-represented families and **collapses to ≈ 0 on singleton families** —
exactly the case that matters when starting a program against a novel target.

## Configs

### `redundancy` (18,759 rows, default)
PDBbind-derived protein–ligand complexes with experimental affinity, annotated with
protein-family size so accuracy can be stratified by redundancy.

| column | description |
|---|---|
| `id` | PDB code |
| `sequence` | protein chain (single-letter) |
| `smiles` | ligand SMILES (from the deposited structure) |
| `pK` | experimental −log10(Kd/Ki/IC50); **higher = stronger** |
| `affinity_type` | `Kd` / `Ki` / `IC50` |
| `year` | PDB deposition year |
| `family_id` | MMseqs2 30%-identity cluster representative |
| `family_size` | number of complexes in that family (**the redundancy axis**) |
| `redundancy_bin` | `1`, `2-5`, `6-20`, `21-80`, `81-300`, `301+` |
| `ligand_nn_tanimoto` | ECFP4 nearest-neighbour similarity to any other ligand |
| `in_core` | member of the balanced 3,360-target quick-eval subset |

### `temporal` (649 rows)
One structurally novel, post-2021-09-30 target (**OpenBind EV-A71 2A protease**), with
measured KD and shipped baselines. Tests within-target ranking on data that no model with
a ≤ 2021 cutoff could have seen.

| column | description |
|---|---|
| `id` | Fragalysis code |
| `sequence` | target protein |
| `smiles` | ligand SMILES |
| `pKD` | experimental −log10(KD); higher = stronger |
| `baseline_molecular_weight`, `baseline_clogp`, `ref_boltz2` | reference predictions |

## Usage

```python
import fambench

def my_model(sequence, smiles):
    return score               # higher = stronger binder

print(fambench.run_redundancy(my_model).summary())
print(fambench.run_temporal(my_model).summary())
```

See <https://github.com/DeepBioScientific/fambench> for the harness, baselines, and CLI.

## Provenance & licensing

- **Redundancy set** is derived from **PDBbind** (protein sequences from the PDB; SMILES
  from deposited ligands; affinities from PDBbind's curated literature values). Only derived
  annotations are redistributed here, under MIT. Users should observe PDBbind's own terms
  when using the underlying structures.
- **Temporal set** is derived from the **OpenBind A71EV2A** release (CC0) and its public
  benchmark repository.

Family sizes are computed with MMseqs2 at 30% sequence identity / 80% coverage.
