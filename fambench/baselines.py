"""Trivial baselines. A model that does not beat molecular weight on novel targets
is not adding value -- this is the whole point of the benchmark.

Predictions follow the FamBench convention: higher = stronger binder.
"""
from __future__ import annotations

import numpy as np


def _mol(smiles):
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    return Chem.MolFromSmiles(smiles)


def molecular_weight(sequence: str, smiles: str) -> float:
    """Ligand molecular weight (proxy for binding; higher MW ~ stronger)."""
    from rdkit.Chem import Descriptors

    m = _mol(smiles)
    return float(Descriptors.MolWt(m)) if m is not None else float("nan")


def clogp(sequence: str, smiles: str) -> float:
    """Ligand cLogP (lipophilicity)."""
    from rdkit.Chem import Crippen

    m = _mol(smiles)
    return float(Crippen.MolLogP(m)) if m is not None else float("nan")


def constant(sequence: str, smiles: str) -> float:
    """A no-information model (sanity floor: should score ~0)."""
    return 0.0


BASELINES = {
    "molecular_weight": molecular_weight,
    "clogp": clogp,
}


def predict_baseline(name, df):
    fn = BASELINES[name]
    return np.array([fn(r.sequence, r.smiles) for r in df.itertuples()], float)
