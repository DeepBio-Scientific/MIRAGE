"""The headline number: the family generalization gap G_m, per method.

    G_m = r(S_f >= 301) - r(S_f = 1)

computed inside the matched pK window (4.5-8.0) so that label spread, which attenuates
correlation, is held constant across the two strata. The CI comes from a two-level
bootstrap -- resample families, then ligands within the resampled families -- because
complexes in one family are not independent draws.

Also reports the family-mean predictor: cross-validated, it uses no ligand information at
all, so its correlation is the memorization ceiling any model is being compared against.

    python scripts/gap_analysis.py

Prediction CSVs (``id,prediction``) are read from paper/predictions; set
MIRAGE_PREDICTIONS / MIRAGE_RESULTS to run the same analysis over your own models. The
molecular-weight and clogp columns are computed on the fly from SMILES (needs rdkit) and
need no CSV.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parent.parent
RED = ROOT / "data" / "redundancy.parquet"
PRED = Path(os.environ.get("MIRAGE_PREDICTIONS", ROOT / "paper" / "predictions"))
OUT = Path(os.environ.get("MIRAGE_RESULTS", ROOT / "paper" / "results")) / "paper_gap_analysis.json"

MODELS = {
    "Nesso-1": "nesso1_redundancy.csv",
    "Boltz-2": "boltz2_redundancy.csv",
    "RF-QSAR": "rf_qsar_redundancy.csv",
    "ligand-kNN": "ligand_knn_redundancy.csv",
}
MATCHED = (4.5, 8.0)
N_BOOT = 500


def family_bin(s: int) -> str:
    if s == 1:
        return "1"
    if s <= 5:
        return "2-5"
    if s <= 20:
        return "6-20"
    if s <= 80:
        return "21-80"
    if s <= 300:
        return "81-300"
    return "301+"


def load_corpus() -> pd.DataFrame:
    red = pd.read_parquet(RED).reset_index(drop=True).rename(columns={"id": "code"})
    red["fbin"] = red.family_size.map(family_bin)
    return red


def load_predictions(red: pd.DataFrame) -> dict[str, dict[str, float]]:
    """One {code: score} map per method, with the trivial baselines computed here."""
    preds = {}
    for name, fname in MODELS.items():
        path = PRED / fname
        if not path.exists():
            print(f"  skipping {name}: {path} not found")
            continue
        d = pd.read_csv(path).rename(columns={"id": "code"}).dropna(subset=["prediction"])
        preds[name] = dict(zip(d.code, d.prediction))
    try:
        from mirage.baselines import predict_baseline
    except ImportError:  # rdkit missing -- the model arms still run
        return preds
    for name, key in (("mol.weight", "molecular_weight"), ("clogp", "clogp")):
        v = predict_baseline(key, red)
        preds[name] = {c: float(x) for c, x in zip(red.code, v) if np.isfinite(x)}
    return preds


def prep(df: pd.DataFrame, pred: dict[str, float], b: str):
    """Labels, scores, and family index groups for one support bin, matched on pK."""
    lo, hi = MATCHED
    d = df[(df.fbin == b) & (df.pK >= lo) & (df.pK <= hi) & (df.code.isin(pred))]
    y = d.pK.values
    p = np.array([pred[c] for c in d.code])
    fam2idx: dict[str, list[int]] = {}
    for i, f in enumerate(d.family_id.values):
        fam2idx.setdefault(f, []).append(i)
    return y, p, [np.array(v) for v in fam2idx.values()]


def r_of(y, p, idx):
    yy, pp = y[idx], p[idx]
    return stats.pearsonr(pp, yy)[0] if yy.std() > 0 and pp.std() > 0 else np.nan


def boot_gap(df, pred, n=N_BOOT, seed=0):
    """Two-level bootstrap CI on G_m: resample families, then ligands within them."""
    rng = np.random.default_rng(seed)
    yh, ph, gh = prep(df, pred, "301+")
    yl, pl, gl = prep(df, pred, "1")

    def one(y, p, groups):
        pick = rng.integers(0, len(groups), len(groups))
        idx = np.concatenate([groups[k] for k in pick])
        idx = idx[rng.integers(0, len(idx), len(idx))]  # second level: ligands
        return r_of(y, p, idx)

    g = []
    for _ in range(n):
        a, b = one(yh, ph, gh), one(yl, pl, gl)
        if np.isfinite(a) and np.isfinite(b):
            g.append(a - b)
    return np.percentile(g, [2.5, 97.5])


def matched_r(df, pred, b):
    y, p, _ = prep(df, pred, b)
    return r_of(y, p, np.arange(len(y)))


def family_mean_ceiling(red: pd.DataFrame) -> float:
    """Cross-validated family-mean predictor: no ligand information whatsoever."""
    y, fid = red.pK.values, red.family_id.values
    oof = np.full(len(y), np.nan)
    for tr, te in KFold(5, shuffle=True, random_state=0).split(y):
        fm = pd.Series(y[tr]).groupby(fid[tr]).mean()
        gm = y[tr].mean()
        oof[te] = [fm.get(fid[i], gm) for i in te]
    return float(stats.pearsonr(oof, y)[0])


def main():
    red = load_corpus()
    preds = load_predictions(red)

    G = {}
    for m, pred in preds.items():
        rh, rl = matched_r(red, pred, "301+"), matched_r(red, pred, "1")
        ci = boot_gap(red, pred)
        G[m] = dict(G=float(rh - rl), r_high=float(rh), r_low=float(rl),
                    ci=[float(ci[0]), float(ci[1])])
        print(f"{m:12s} G={rh - rl:+.3f} [r(301+)={rh:+.3f} r(1)={rl:+.3f}] "
              f"95%CI[{ci[0]:+.2f},{ci[1]:+.2f}]", flush=True)

    rfm = family_mean_ceiling(red)
    print(f"family-mean random: {rfm:+.3f}", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"G": G, "family_mean_random": rfm}, indent=1))
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
