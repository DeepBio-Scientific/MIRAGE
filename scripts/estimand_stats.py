"""Estimand, matched-sample controls, and cross-fitted calibration.

Three checks a reviewer asked for that the shipped analysis could not answer:

1. What G_m is actually made of. Pearson r in the S_f=1 stratum is purely a
   cross-target calibration statistic (every family contributes one complex), while in
   the S_f>=301 stratum it mixes between-family calibration with within-family ligand
   ranking. This decomposes the high-support correlation into those parts and reports the
   within-target ranking statistic a discovery programme actually cares about.

2. Controls on the co-folder's own complexes. The shipped controls are scored on all
   18,759 complexes while Nesso-1 covers 3,289 and Boltz-2 583, so control-vs-co-folder
   gaps are computed on different samples. This recomputes every control restricted to
   each co-folder's exact coverage and reports paired Delta-G with a shared bootstrap.

3. Cross-fitted calibration. |error| requires putting an arbitrary-scale score (ipTM,
   smina, molecular weight) on the pK scale. The shipped code fits one global linear
   calibration on the whole evaluated sample, so a dense family helps calibrate its own
   errors. This refits calibration out-of-fold with GroupKFold on family and recomputes
   the support slope.

    python scripts/estimand_stats.py

Prediction CSVs (``id,prediction``) are read from paper/predictions; set
MIRAGE_PREDICTIONS / MIRAGE_RESULTS to run the same analysis over your own models.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parent.parent
PRED = Path(os.environ.get("MIRAGE_PREDICTIONS", ROOT / "paper" / "predictions"))
OUT = Path(os.environ.get("MIRAGE_RESULTS", ROOT / "paper" / "results")) / "estimand_stats.json"
BINS = ["1", "2-5", "6-20", "21-80", "81-300", "301+"]
MODELS = {"Nesso-1": "nesso1_redundancy.csv", "Boltz-2": "boltz2_redundancy.csv",
          "Chai-1 (ipTM)": "chai1_iptm_redundancy.csv", "gnina": "gnina_affinity_redundancy.csv",
          "smina": "smina_affinity_redundancy.csv", "RF-QSAR": "rf_qsar_redundancy.csv",
          "ligand-kNN": "ligand_knn_redundancy.csv"}


def fbin(s):
    return ("1" if s == 1 else "2-5" if s <= 5 else "6-20" if s <= 20
            else "21-80" if s <= 80 else "81-300" if s <= 300 else "301+")


def load(name, red):
    d = pd.read_csv(PRED / MODELS[name]).rename(columns={"id": "code"})
    m = red.merge(d, on="code").dropna(subset=["prediction"])
    return m[m.pK.between(4.5, 8.0)]


def r_of(y, p):
    if len(y) < 4 or np.std(y) == 0 or np.std(p) == 0:
        return float("nan")
    return float(stats.pearsonr(p, y)[0])


def groups_of(fam):
    idx = {}
    for i, f in enumerate(fam):
        idx.setdefault(f, []).append(i)
    return [np.asarray(v) for v in idx.values()]


def boot_idx(groups, rng):
    pick = rng.integers(0, len(groups), len(groups))
    i = np.concatenate([groups[k] for k in pick])
    return i[rng.integers(0, len(i), len(i))]


# ---------------------------------------------------------------- 1. estimand
def decompose(red) -> dict:
    print("=" * 92)
    print("1. What the stratum correlation is made of (between-family vs within-family)")
    print("=" * 92)
    out = {}
    for name in ["Nesso-1", "Boltz-2", "RF-QSAR", "smina"]:
        m = load(name, red)
        m = m.assign(fb=m.family_size.map(fbin))
        print(f"\n  {name}")
        print(f"    {'bin':7s} {'n':>5s} {'fams':>5s} {'r total':>9s} {'r between':>10s} "
              f"{'r within':>9s} {'n usable':>9s}")
        rows = {}
        for b in BINS:
            s = m[m.fb == b]
            if len(s) < 4:
                continue
            tot = r_of(s.pK.values, s.prediction.values)
            g = s.groupby("family_id")
            fm = g.agg(y=("pK", "mean"), p=("prediction", "mean"))
            between = r_of(fm.y.values, fm.p.values) if len(fm) >= 4 else float("nan")
            multi = s[s.family_id.isin(g.size()[g.size() > 1].index)]
            if len(multi) >= 4:
                yc = multi.pK - multi.groupby("family_id").pK.transform("mean")
                pc = multi.prediction - multi.groupby("family_id").prediction.transform("mean")
                within = r_of(yc.values, pc.values)
            else:
                within = float("nan")
            rows[b] = dict(n=int(len(s)), n_families=int(s.family_id.nunique()),
                           r_total=tot, r_between=between, r_within=within,
                           n_within_usable=int(len(multi)))
            print(f"    {b:7s} {len(s):5d} {s.family_id.nunique():5d} {tot:+9.3f} "
                  f"{between:+10.3f} {within:+9.3f} {len(multi):9d}")
        out[name] = rows
    return out


def within_target(red) -> dict:
    """Ranking within an exact protein sequence -- the discovery-programme statistic."""
    print("\n" + "=" * 92)
    print("2. Within-target ranking (exact identical sequence, >=5 ligands): mean Spearman")
    print("=" * 92)
    out = {}
    print(f"  {'model':12s} {'stratum':16s} {'targets':>8s} {'ligands':>8s} "
          f"{'mean rho':>9s} {'median':>8s}")
    for name in ["Nesso-1", "Boltz-2", "RF-QSAR", "ligand-kNN", "smina", "gnina"]:
        m = load(name, red)
        m = m.assign(fb=m.family_size.map(fbin))
        out[name] = {}
        for lab, sel in [("novel S_f<=5", m[m.family_size <= 5]),
                         ("redundant S_f>=301", m[m.family_size >= 301])]:
            rhos, nlig = [], 0
            for _, g in sel.groupby("sequence"):
                if len(g) >= 5 and g.pK.std() > 0 and g.prediction.std() > 0:
                    rhos.append(stats.spearmanr(g.prediction, g.pK)[0])
                    nlig += len(g)
            if rhos:
                out[name][lab] = dict(n_targets=len(rhos), n_ligands=nlig,
                                      mean_rho=float(np.nanmean(rhos)),
                                      median_rho=float(np.nanmedian(rhos)))
                print(f"  {name:12s} {lab:16s} {len(rhos):8d} {nlig:8d} "
                      f"{np.nanmean(rhos):+9.3f} {np.nanmedian(rhos):+8.3f}")
            else:
                print(f"  {name:12s} {lab:16s} {'0':>8s} {'-':>8s} {'n/a':>9s} {'n/a':>8s}")
    return out


# ------------------------------------------------- 3. matched-sample controls
def matched_controls(red) -> dict:
    print("\n" + "=" * 92)
    print("3. Controls recomputed on each co-folder's exact complexes, paired Delta-G")
    print("=" * 92)
    out = {}
    for cof in ["Nesso-1", "Boltz-2"]:
        base = load(cof, red)
        ids = set(base.code)
        print(f"\n  --- restricted to {cof}'s {len(ids)} complexes ---")
        print(f"    {'method':12s} {'n':>6s} {'G_m':>7s} "
              f"{'delta vs cofolder':>19s} {'95% CI':>18s}")
        lo_b = base[base.family_size == 1]
        hi_b = base[base.family_size >= 301]
        g_cof = (r_of(hi_b.pK.values, hi_b.prediction.values)
                 - r_of(lo_b.pK.values, lo_b.prediction.values))
        out[cof] = {cof: dict(n=int(len(base)), G=float(g_cof), delta=0.0, ci=[0.0, 0.0])}
        print(f"    {cof:12s} {len(base):6d} {g_cof:+7.3f} {'-':>19s} {'-':>18s}")
        for ctrl in ["RF-QSAR", "ligand-kNN", "smina", "gnina"]:
            c = load(ctrl, red)
            c = c[c.code.isin(ids)][["code", "prediction"]]
            j = base.merge(c, on="code", suffixes=("_cof", "_ctrl"))
            lo, hi = j[j.family_size == 1], j[j.family_size >= 301]
            if len(lo) < 10 or len(hi) < 10:
                print(f"    {ctrl:12s} {len(j):6d} underpowered")
                continue
            g_ctrl = r_of(hi.pK.values, hi.prediction_ctrl.values) - \
                r_of(lo.pK.values, lo.prediction_ctrl.values)
            g_c = r_of(hi.pK.values, hi.prediction_cof.values) - \
                r_of(lo.pK.values, lo.prediction_cof.values)
            rng = np.random.default_rng(0)
            gl, gh = groups_of(lo.family_id.values), groups_of(hi.family_id.values)
            yl, yh = lo.pK.values, hi.pK.values
            draws = []
            for _ in range(2000):
                il, ih = boot_idx(gl, rng), boot_idx(gh, rng)
                a = (r_of(yh[ih], hi.prediction_cof.values[ih])
                     - r_of(yl[il], lo.prediction_cof.values[il]))
                b = (r_of(yh[ih], hi.prediction_ctrl.values[ih])
                     - r_of(yl[il], lo.prediction_ctrl.values[il]))
                if np.isfinite(a) and np.isfinite(b):
                    draws.append(a - b)
            ci = np.percentile(draws, [2.5, 97.5])
            out[cof][ctrl] = dict(n=int(len(j)), G=float(g_ctrl), delta=float(g_c - g_ctrl),
                                  ci=[float(ci[0]), float(ci[1])])
            print(f"    {ctrl:12s} {len(j):6d} {g_ctrl:+7.3f} {g_c - g_ctrl:+19.3f} "
                  f"{f'[{ci[0]:+.2f},{ci[1]:+.2f}]':>18s}")
    return out


# --------------------------------------------- 4. cross-fitted calibration
def calib_global(p, y):
    X = np.c_[np.ones_like(p), p]
    return X @ np.linalg.lstsq(X, y, rcond=None)[0]


def calib_crossfit(p, y, fam, n_splits=5):
    """Out-of-fold linear calibration, folds disjoint by family."""
    out = np.full(len(y), np.nan)
    n_splits = min(n_splits, len(np.unique(fam)))
    for tr, te in GroupKFold(n_splits=n_splits).split(p, y, groups=fam):
        X = np.c_[np.ones(len(tr)), p[tr]]
        b = np.linalg.lstsq(X, y[tr], rcond=None)[0]
        out[te] = b[0] + b[1] * p[te]
    return out


def ols_cr1(X, y, groups):
    XtX = np.linalg.pinv(X.T @ X)
    b = XtX @ X.T @ y
    e = y - X @ b
    k = X.shape[1]
    meat = np.zeros((k, k))
    g = pd.Series(groups)
    for idx in g.groupby(g).indices.values():
        s = X[idx].T @ e[idx]
        meat += np.outer(s, s)
    ng = g.nunique()
    sc = (ng / max(ng - 1, 1)) * ((len(y) - 1) / max(len(y) - k, 1))
    return b, np.sqrt(np.diag(XtX @ (sc * meat) @ XtX)), ng


def calibration_check(red) -> dict:
    print("\n" + "=" * 92)
    print("4. Support slope under global vs family-disjoint cross-fitted calibration")
    print("=" * 92)
    print(f"  {'model':14s} {'coef global':>12s} {'t':>7s} {'coef xfit':>11s} {'t':>7s}")
    out = {}
    full = red  # slopes use the full pK range, as in the shipped harness
    for name in MODELS:
        d = pd.read_csv(PRED / MODELS[name]).rename(columns={"id": "code"})
        m = full.merge(d, on="code").dropna(subset=["prediction"])
        y = m.pK.values
        p = m.prediction.values.astype(float)
        fam = m.family_id.values
        res = {}
        for tag, cal in [("global", calib_global(p, y)),
                         ("xfit", calib_crossfit(p, y, fam))]:
            ok = np.isfinite(cal)
            ae = np.abs(cal[ok] - y[ok])
            mm = m[ok]
            X = np.column_stack([
                np.ones(ok.sum()), np.log10(mm.family_size.values),
                mm.ligand_nn_tanimoto.values, np.log10(mm.sequence.str.len().values),
                (mm.year.values - 2000) / 10.0, mm.famvar.values,
                (mm.affinity_type.values == "Ki").astype(float),
                (mm.affinity_type.values == "IC50").astype(float), y[ok], y[ok] ** 2])
            b, se, _ = ols_cr1(X, ae, mm.family_id.values)
            res[tag] = dict(coef=float(b[1]), t=float(b[1] / se[1]))
        out[name] = res
        print(f"  {name:14s} {res['global']['coef']:+12.3f} {res['global']['t']:+7.2f} "
              f"{res['xfit']['coef']:+11.3f} {res['xfit']['t']:+7.2f}")
    return out


def main():
    red = pd.read_parquet(ROOT / "data" / "redundancy.parquet").rename(columns={"id": "code"})
    red = red.join(red.groupby("family_id").pK.var().rename("famvar"), on="family_id")
    red["famvar"] = red.famvar.fillna(0.0)
    res = {"decomposition": decompose(red), "within_target": within_target(red),
           "matched_controls": matched_controls(red), "calibration": calibration_check(red)}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
