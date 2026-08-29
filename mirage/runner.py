"""High-level runner: plug in a model, get a report.

A model is any callable ``predict(sequence: str, smiles: str) -> float`` returning a
score where higher means stronger binding. If your model returns log10(IC50) or another
"lower is stronger" score, pass ``higher_is_stronger=False`` and MIRAGE flips the sign.

You can also skip Python entirely and score a predictions CSV -- see ``score_csv``.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from . import baselines as _bl
from .data import load
from .evaluate import evaluate_redundancy, evaluate_temporal


def _apply(model, df):
    out = np.empty(len(df))
    for i, r in enumerate(df.itertuples()):
        try:
            out[i] = float(model(r.sequence, r.smiles))
        except Exception:
            out[i] = np.nan
    return out


def run_redundancy(model: Callable[[str, str], float], *, core: bool = True,
                   higher_is_stronger: bool = True, path: str | None = None,
                   progress: bool = True):
    """Evaluate a model on the redundancy set. ``core=True`` uses the balanced
    3,360-target subset (recommended for a first pass)."""
    df = load("redundancy", path=path, core=core)
    if progress:
        print(f"[mirage] scoring {len(df)} complexes "
              f"({'core' if core else 'full'} redundancy set)...")
    preds = _apply(model, df)
    return evaluate_redundancy(
        df.pK.values, preds, df.family_size.values,
        affinity_type=df.affinity_type.values,
        higher_is_stronger=higher_is_stronger,
    )


def run_temporal(model: Callable[[str, str], float], *,
                 higher_is_stronger: bool = True, path: str | None = None,
                 progress: bool = True):
    """Evaluate a model on the novel-target temporal set, against the shipped baselines."""
    df = load("temporal", path=path)
    if progress:
        print(f"[mirage] scoring {len(df)} compounds (novel target)...")
    preds = _apply(model, df)
    bl = {"molecular_weight": df.baseline_molecular_weight.values,
          "clogp": df.baseline_clogp.values}
    if "ref_boltz2" in df.columns:
        bl["Boltz-2 (reference)"] = df.ref_boltz2.values
    return evaluate_temporal(df.pKD.values, preds, baselines=bl,
                             higher_is_stronger=higher_is_stronger)


def score_csv(name: str, pred_csv: str, *, id_col: str = "id",
              pred_col: str = "prediction", higher_is_stronger: bool = True,
              path: str | None = None):
    """Score a predictions CSV (columns: id, prediction) against the named set.

    This is the language-agnostic path: produce predictions however you like, write a
    two-column CSV, and score it.
    """
    import pandas as pd

    df = load("redundancy" if name == "redundancy" else "temporal", path=path)
    pr = pd.read_csv(pred_csv).set_index(id_col)[pred_col]
    preds = df["id"].map(pr).values.astype(float)
    n_missing = int(np.isnan(preds).sum())
    if n_missing:
        print(f"[mirage] warning: {n_missing}/{len(df)} ids missing from predictions")
    if name == "redundancy":
        return evaluate_redundancy(df.pK.values, preds, df.family_size.values,
                                   affinity_type=df.affinity_type.values,
                                   higher_is_stronger=higher_is_stronger)
    bl = {"molecular_weight": df.baseline_molecular_weight.values,
          "clogp": df.baseline_clogp.values}
    return evaluate_temporal(df.pKD.values, preds, baselines=bl,
                             higher_is_stronger=higher_is_stronger)


def baseline_model(name: str):
    """Return a shipped baseline as a plug-in model callable."""
    return _bl.BASELINES[name]
