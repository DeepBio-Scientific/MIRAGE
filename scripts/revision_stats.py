"""Reviewer-requested inference: family-clustered SEs, assay-type sensitivity,
ligand-axis interaction, and support-annotation breakdown.

Every number this prints is used in the revision of the manuscript. Run:

    python scripts/revision_stats.py            # prints tables, writes JSON

Outputs paper/results/revision_stats.json. Prediction CSVs (``id,prediction``) are read
from paper/predictions; set MIRAGE_PREDICTIONS / MIRAGE_RESULTS to run the same analysis
over your own models.

Why this exists: the submitted covariate regression reported ordinary OLS
t-statistics while the rest of the paper (correctly) treats complexes within a
family as dependent. This script re-does that inference with CR1 cluster-robust
standard errors clustered on family_id, and adds the sensitivity analyses a
reviewer needs to see: G_m by assay type, and the ligand-similarity axis with
two-level bootstrap CIs and an explicit S_f x ligand interaction term.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
RED = ROOT / "data" / "redundancy.parquet"
EXPO = ROOT / "data" / "exposure_annotations.parquet"
PRED = Path(os.environ.get("MIRAGE_PREDICTIONS", ROOT / "paper" / "predictions"))
OUT = Path(os.environ.get("MIRAGE_RESULTS", ROOT / "paper" / "results")) / "revision_stats.json"

MODELS = {
    "Nesso-1": "nesso1_redundancy.csv",
    "Boltz-2": "boltz2_redundancy.csv",
    "Chai-1 (ipTM)": "chai1_iptm_redundancy.csv",
    "ESMFold2 (ipTM)": "esmfold2_iptm_redundancy.csv",
    "gnina": "gnina_affinity_redundancy.csv",
    "smina": "smina_affinity_redundancy.csv",
    "RF-QSAR": "rf_qsar_redundancy.csv",
    "ligand-kNN": "ligand_knn_redundancy.csv",
}
# Ligand-similarity strata used in Table 4 of the submission.
LIG_CUTS = (0.50, 0.75)
REDUNDANCY_BINS = ["1", "2-5", "6-20", "21-80", "81-300", "301+"]
MATCHED = (4.5, 8.0)


def load_corpus() -> pd.DataFrame:
    red = pd.read_parquet(RED).rename(columns={"id": "code"})
    red["loglen"] = np.log10(red.sequence.str.len())
    red = red.join(red.groupby("family_id").pK.var().rename("famvar"), on="family_id")
    red["famvar"] = red.famvar.fillna(0.0)
    if EXPO.exists():
        # exact-target / near-identical exposure, separated from family support
        red = red.merge(pd.read_parquet(EXPO).rename(columns={"id": "code"}), on="code")
    else:
        red["n_exact_sib"] = red.sequence.map(red.groupby("sequence").size()) - 1
    red["E_exact"] = (red.n_exact_sib > 0).astype(float)
    return red


def load(red: pd.DataFrame, name: str) -> pd.DataFrame:
    d = pd.read_csv(PRED / MODELS[name]).rename(columns={"id": "code"})
    return red.merge(d, on="code", how="inner").dropna(subset=["prediction"])


def calibrated_abs_error(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    """|error| after linearly calibrating the score onto the label scale."""
    X = np.c_[np.ones_like(p), p]
    return np.abs(X @ np.linalg.lstsq(X, y, rcond=None)[0] - y)


def ols_cr1(X: np.ndarray, y: np.ndarray, groups: np.ndarray):
    """OLS coefficients with classical and CR1 cluster-robust standard errors."""
    XtX_inv = np.linalg.pinv(X.T @ X)
    b = XtX_inv @ X.T @ y
    e = y - X @ b
    n, k = X.shape
    se_ols = np.sqrt(np.diag(XtX_inv) * (e @ e) / max(n - k, 1))
    meat = np.zeros((k, k))
    g = pd.Series(groups)
    for idx in g.groupby(g).indices.values():
        s = X[idx].T @ e[idx]
        meat += np.outer(s, s)
    n_g = g.nunique()
    scale = (n_g / max(n_g - 1, 1)) * ((n - 1) / max(n - k, 1))
    se_cr1 = np.sqrt(np.diag(XtX_inv @ (scale * meat) @ XtX_inv))
    return b, se_ols, se_cr1, n_g


def design(m: pd.DataFrame, interaction: bool = False):
    y = m.pK.values
    ae = calibrated_abs_error(m.prediction.values.astype(float), y)
    log_sf, lig = np.log10(m.family_size.values), m.ligand_nn_tanimoto.values
    cols = [np.ones(len(m)), log_sf, lig, m.loglen.values,
            (m.year.values - 2000) / 10.0, m.famvar.values,
            (m.affinity_type.values == "Ki").astype(float),
            (m.affinity_type.values == "IC50").astype(float), y, y ** 2]
    names = ["const", "log10_Sf", "lig_tanimoto", "log10_len", "year_decade",
             "family_pK_var", "is_Ki", "is_IC50", "pK", "pK2"]
    if interaction:
        cols.append(log_sf * lig)
        names.append("log10_Sf_x_lig")
    return np.column_stack(cols), ae, names


def matched(m: pd.DataFrame) -> pd.DataFrame:
    return m[(m.pK >= MATCHED[0]) & (m.pK <= MATCHED[1])]


def pearson(sub: pd.DataFrame) -> float:
    if len(sub) < 4 or sub.pK.std() == 0 or sub.prediction.std() == 0:
        return float("nan")
    return float(stats.pearsonr(sub.prediction.astype(float), sub.pK)[0])


def two_level_bootstrap(sub: pd.DataFrame, n: int = 2000, seed: int = 0) -> np.ndarray:
    """Resample families with replacement, then rows within the draw."""
    rng = np.random.default_rng(seed)
    s = sub.reset_index(drop=True)
    groups = [np.asarray(v) for v in s.groupby("family_id").indices.values()]
    y, p = s.pK.values, s.prediction.values.astype(float)
    out = []
    for _ in range(n):
        pick = rng.integers(0, len(groups), len(groups))
        idx = np.concatenate([groups[k] for k in pick])
        idx = idx[rng.integers(0, len(idx), len(idx))]
        if y[idx].std() > 0 and p[idx].std() > 0:
            out.append(stats.pearsonr(p[idx], y[idx])[0])
    return np.array(out)


def main() -> None:
    red = load_corpus()
    res: dict = {}

    print("=" * 84)
    print("1. Covariate regression: |error| ~ log10 S_f + covariates, CR1 clustered on family")
    print("=" * 84)
    print(f"{'method':16s} {'n':>6s} {'families':>9s} {'coef':>8s} "
          f"{'t_OLS':>8s} {'t_cluster':>10s}")
    res["covariate_regression"] = {}
    for name in MODELS:
        m = load(red, name)
        X, ae, names = design(m)
        b, se_o, se_c, n_g = ols_cr1(X, ae, m.family_id.values)
        j = names.index("log10_Sf")
        res["covariate_regression"][name] = dict(
            n=int(len(m)), n_families=int(n_g), coef=float(b[j]),
            t_ols=float(b[j] / se_o[j]), t_cluster=float(b[j] / se_c[j]))
        print(f"{name:16s} {len(m):6d} {n_g:9d} {b[j]:+8.3f} {b[j]/se_o[j]:+8.1f} "
              f"{b[j]/se_c[j]:+10.1f}")

    print("\n" + "=" * 84)
    print("2. Family generalization gap by assay type (matched pK)")
    print("=" * 84)
    res["assay_sensitivity"] = {}
    for name in ["Nesso-1", "Boltz-2", "RF-QSAR", "ligand-kNN"]:
        mm = matched(load(red, name))
        res["assay_sensitivity"][name] = {}
        print(f"\n{name}")
        for lab, sel in [("all", mm), ("Kd", mm[mm.affinity_type == "Kd"]),
                         ("Ki", mm[mm.affinity_type == "Ki"]),
                         ("Kd+Ki", mm[mm.affinity_type.isin(["Kd", "Ki"])]),
                         ("IC50", mm[mm.affinity_type == "IC50"])]:
            lo, hi = sel[sel.family_size == 1], sel[sel.family_size >= 301]
            rl, rh = pearson(lo), pearson(hi)
            res["assay_sensitivity"][name][lab] = dict(
                G=float(rh - rl), r_singleton=float(rl), r_redundant=float(rh),
                n_singleton=int(len(lo)), n_redundant=int(len(hi)))
            print(f"  {lab:6s} G={rh-rl:+.2f}  r(S_f=1)={rl:+.2f} (n={len(lo)})  "
                  f"r(S_f>=301)={rh:+.2f} (n={len(hi)})")

    print("\n" + "=" * 84)
    print("3. Ligand axis: Table 4 cells with two-level bootstrap CIs, and the interaction")
    print("=" * 84)
    m = matched(load(red, "Nesso-1")).reset_index(drop=True)
    fam = np.where(m.family_size <= 5, "novel (1-5)",
                   np.where(m.family_size <= 80, "mid (6-80)", "redundant (81+)"))
    lig = np.where(m.ligand_nn_tanimoto <= LIG_CUTS[0], "novel ligand",
                   np.where(m.ligand_nn_tanimoto <= LIG_CUTS[1], "mid", "similar ligand"))
    m["fam_stratum"], m["lig_stratum"] = fam, lig
    grid, draws = {}, {}
    for f in ["novel (1-5)", "mid (6-80)", "redundant (81+)"]:
        row = []
        for l in ["novel ligand", "mid", "similar ligand"]:
            sub = m[(m.fam_stratum == f) & (m.lig_stratum == l)]
            d = two_level_bootstrap(sub)
            draws[(f, l)] = d
            lo, hi = np.percentile(d, [2.5, 97.5])
            grid[f"{f} | {l}"] = dict(r=pearson(sub), n=int(len(sub)),
                                      ci=[float(lo), float(hi)],
                                      label_sd=float(sub.pK.std()),
                                      n_families=int(sub.family_id.nunique()))
            row.append(f"{pearson(sub):+.2f} [{lo:+.2f},{hi:+.2f}] "
                       f"(n={len(sub)}, sd={sub.pK.std():.2f})")
        print(f"  {f:17s} " + "  ".join(f"{c:26s}" for c in row))
    res["ligand_grid"] = grid

    diffs = {}
    for f in ["novel (1-5)", "mid (6-80)", "redundant (81+)"]:
        d = draws[(f, "novel ligand")] - draws[(f, "similar ligand")]
        lo, hi = np.percentile(d, [2.5, 97.5])
        diffs[f] = dict(delta=float(d.mean()), ci=[float(lo), float(hi)],
                        p_gt_zero=float((d > 0).mean()))
        print(f"  {f:17s} r(novel lig) - r(similar lig) = {d.mean():+.2f} "
              f"[{lo:+.2f},{hi:+.2f}]  P(>0)={float((d>0).mean()):.2f}")
    res["ligand_axis_contrast"] = diffs

    X, ae, names = design(m, interaction=True)
    b, se_o, se_c, n_g = ols_cr1(X, ae, m.family_id.values)
    res["interaction_model"] = {}
    for key in ["log10_Sf", "lig_tanimoto", "log10_Sf_x_lig"]:
        j = names.index(key)
        res["interaction_model"][key] = dict(coef=float(b[j]),
                                             t_cluster=float(b[j] / se_c[j]))
        print(f"  {key:16s} coef {b[j]:+.4f}  t_cluster {b[j]/se_c[j]:+.2f}")

    print("\n" + "=" * 84)
    print("4. What S_f counts: support excluding the query, and exact-target overlap")
    print("=" * 84)
    dup = red.sequence.map(red.groupby("sequence").size())
    ann = red.assign(has_identical_sibling=(dup > 1)).groupby("redundancy_bin").agg(
        n=("code", "size"), median_support_excl_self=("family_size", lambda s: (s - 1).median()),
        frac_identical_sequence_sibling=("has_identical_sibling", "mean"))
    ann = ann.reindex(["1", "2-5", "6-20", "21-80", "81-300", "301+"])
    print(ann.to_string(float_format=lambda v: f"{v:.3f}"))
    res["support_annotation"] = json.loads(ann.to_json(orient="index"))
    res["support_annotation_notes"] = dict(
        family_size_includes_query=True,
        support_excluding_query="family_size - 1",
        n_complexes=int(len(red)),
        deposition_year_range=[int(red.year.min()), int(red.year.max())],
        all_pre_cutoff=bool(red.year.max() < 2021))

    print("\n" + "=" * 84)
    print("5. Family-mean baseline: train-fold-only vs. label-including (leaky)")
    print("=" * 84)
    from sklearn.model_selection import KFold
    y, fid = red.pK.values, red.family_id.values
    oof = np.full(len(y), np.nan)
    for tr, te in KFold(5, shuffle=True, random_state=0).split(y):
        fm = pd.Series(y[tr]).groupby(fid[tr]).mean()
        gm = y[tr].mean()
        oof[te] = [fm.get(fid[i], gm) for i in te]
    r_oof = float(stats.pearsonr(oof, y)[0])
    r_leaky = float(stats.pearsonr(pd.Series(y).groupby(fid).transform("mean").values, y)[0])
    res["family_mean_baseline"] = dict(train_fold_only=r_oof, including_own_label=r_leaky)
    print(f"  train-fold-only (as reported): {r_oof:+.3f}")
    print(f"  including own label (leaky, not used): {r_leaky:+.3f}")

    print("\n" + "=" * 84)
    print("6. Exact-target repetition vs. family support")
    print("=" * 84)
    print("6a. does the family-support coefficient survive adding exact-target exposure E?")
    print(f"  {'method':16s} {'coef_Sf':>9s} {'t_Sf':>7s} | {'coef_Sf|E':>10s} {'t_Sf|E':>8s} "
          f"{'coef_E':>8s} {'t_E':>6s}")
    res["exact_exposure_regression"] = {}
    for name in MODELS:
        m = load(red, name)
        X, ae, names = design(m)
        b0, _, sc0, _ = ols_cr1(X, ae, m.family_id.values)
        j = names.index("log10_Sf")
        X2 = np.column_stack([X, m.E_exact.values])
        n2 = names + ["E_exact"]
        b1, _, sc1, _ = ols_cr1(X2, ae, m.family_id.values)
        je, j2 = n2.index("E_exact"), n2.index("log10_Sf")
        res["exact_exposure_regression"][name] = dict(
            coef_Sf=float(b0[j]), t_Sf=float(b0[j] / sc0[j]),
            coef_Sf_given_E=float(b1[j2]), t_Sf_given_E=float(b1[j2] / sc1[j2]),
            coef_E=float(b1[je]), t_E=float(b1[je] / sc1[je]))
        print(f"  {name:16s} {b0[j]:+9.3f} {b0[j]/sc0[j]:+7.2f} | {b1[j2]:+10.3f} "
              f"{b1[j2]/sc1[j2]:+8.2f} {b1[je]:+8.3f} {b1[je]/sc1[je]:+6.2f}")

    print("\n6b. support curve among targets with NO identical-sequence sibling (E=0)")
    res["restricted_support_curve"] = {}
    for name in MODELS:
        m = matched(load(red, name))
        for tag, sub in [("E=0", m[m.E_exact == 0]), ("all", m)]:
            cells = {b: (pearson(sub[sub.redundancy_bin == b]),
                         int((sub.redundancy_bin == b).sum()),
                         int(sub[sub.redundancy_bin == b].family_id.nunique()))
                     for b in REDUNDANCY_BINS}
            lo, hi = sub[sub.redundancy_bin == "1"], sub[sub.redundancy_bin == "301+"]
            G = cells["301+"][0] - cells["1"][0]
            ci = [float("nan")] * 2
            if len(lo) > 10 and len(hi) > 10:
                dl, dh = two_level_bootstrap(lo), two_level_bootstrap(hi)
                k = min(len(dl), len(dh))
                ci = list(np.percentile(dh[:k] - dl[:k], [2.5, 97.5]))
            res["restricted_support_curve"].setdefault(name, {})[tag] = dict(
                cells={b: dict(r=cells[b][0], n=cells[b][1], n_families=cells[b][2])
                       for b in REDUNDANCY_BINS},
                G=float(G), ci=[float(ci[0]), float(ci[1])])
            print(f"  {name:16s} {tag:5s} " +
                  "  ".join(f"{b}:{cells[b][0]:+.2f}(n={cells[b][1]})" for b in REDUNDANCY_BINS))
            print(f"  {'':16s} {'':5s} G_m={G:+.2f} [{ci[0]:+.2f},{ci[1]:+.2f}]")

    print("\n6c. coarse contrast at higher identity stringency (S_f<=5 vs S_f>=81)")
    res["stringency_contrast"] = {}
    for name in ["Nesso-1", "gnina", "smina", "RF-QSAR", "ligand-kNN"]:
        m = matched(load(red, name))
        for lab, col in [("exact", "n_exact_sib"), ("99% id", "n_sib_99"), ("95% id", "n_sib_95")]:
            if col not in m.columns:
                continue
            s_ = m[m[col] == 0]
            nov, rdn = s_[s_.family_size <= 5], s_[s_.family_size >= 81]
            if len(nov) < 15 or len(rdn) < 15:
                print(f"  {name:12s} no sibling at {lab:7s}: n={len(nov)}/{len(rdn)} underpowered")
                continue
            rn, rr = pearson(nov), pearson(rdn)
            dn, dr = two_level_bootstrap(nov), two_level_bootstrap(rdn)
            k = min(len(dn), len(dr))
            ci = np.percentile(dr[:k] - dn[:k], [2.5, 97.5])
            res["stringency_contrast"].setdefault(name, {})[lab] = dict(
                r_novel=rn, n_novel=int(len(nov)), r_redundant=rr, n_redundant=int(len(rdn)),
                delta=float(rr - rn), ci=[float(ci[0]), float(ci[1])])
            print(f"  {name:12s} no sibling at {lab:7s}: novel {rn:+.2f} (n={len(nov)})  "
                  f"redundant {rr:+.2f} (n={len(rdn)})  delta {rr-rn:+.2f} "
                  f"[{ci[0]:+.2f},{ci[1]:+.2f}]")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
