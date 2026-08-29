"""MIRAGE -- Family-Stratified Protein-Ligand Affinity Benchmark.

Frontier co-folding affinity models report strong correlations, but that accuracy is
largely carried by protein families heavily represented in the PDB. MIRAGE measures
how a model's accuracy scales with protein-family redundancy, and how it does on a
genuinely novel target -- the case that matters for a new drug program.

Quick start
-----------
    import mirage

    def my_model(sequence, smiles):
        return score  # higher = stronger binder

    print(mirage.run_redundancy(my_model).summary())
    print(mirage.run_temporal(my_model).summary())
"""
from .data import load
from .evaluate import (
    RedundancyReport,
    TemporalReport,
    evaluate_redundancy,
    evaluate_temporal,
)
from .runner import (
    baseline_model,
    run_redundancy,
    run_temporal,
    score_csv,
)

__version__ = "0.1.0"
__all__ = [
    "load",
    "run_redundancy",
    "run_temporal",
    "score_csv",
    "baseline_model",
    "evaluate_redundancy",
    "evaluate_temporal",
    "RedundancyReport",
    "TemporalReport",
]
