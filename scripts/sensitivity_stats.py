"""Two robustness analyses the benchmark should not ship without.

1. Clustering sensitivity. The whole instrument rests on one choice -- MMseqs2 at 30%
   sequence identity. Re-annotate family support at 20/30/40/50% identity and recompute
   G_m from the *same* released predictions, so the question "is this an artifact of the
   30% threshold?" is answered rather than deferred.

2. Paired ranking contrasts. The novel-family ranking inversion is quoted as bare point
   estimates, and worse, each model is scored on its own coverage (RF-QSAR on 3,230
   complexes, Nesso-1 on 1,095, Boltz-2 on 189). This computes r for both models of a
   pair on the complexes they share, and bootstraps the difference with the same
   two-level (family, then ligand) resampling used for G_m.

    python scripts/sensitivity_stats.py --clusters <dir with c<id>_cluster.tsv files>

If --clusters is omitted the clustering arm is skipped and only the paired contrasts run.
Cluster files are produced by:

    mmseqs easy-cluster all.fasta c0.4 tmp --min-seq-id 0.4 -c 0.8 --cov-mode 0

Prediction CSVs (``id,prediction``) are read from paper/predictions; set
MIRAGE_PREDICTIONS / MIRAGE_RESULTS to run the same analysis over your own models.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
RED = ROOT / "data" / "redundancy.parquet"
PRED = Path(os.environ.get("MIRAGE_PREDICTIONS", ROOT / "paper" / "predictions"))
OUT = Path(os.environ.get("MIRAGE_RESULTS", ROOT / "paper" / "results")) / "sensitivity_stats.json"
BINS = ["1", "2-5", "6-20", "21-80", "81-300", "301+"]
MODELS = {"Nesso-1": "nesso1_redundancy.csv", "Boltz-2": "boltz2_redundancy.csv",
          "gnina": "gnina_affinity_redundancy.csv", "smina": "smina_affinity_redundancy.csv",
          "RF-QSAR": "rf_qsar_redundancy.csv", "ligand-kNN": "ligand_knn_redundancy.csv"}


def family_bin(s: int) -> str:
    return ("1" if s == 1 else "2-5" if s <= 5 else "6-20" if s <= 20
            else "21-80" if s <= 80 else "81-300" if s <= 300 else "301+")


def pearson(y, p) -> float:
    if len(y) < 4 or np.std(y) == 0 or np.std(p) == 0:
        return float("nan")
    return float(stats.pearsonr(p, y)[0])


def family_groups(fam: np.ndarray) -> list[np.ndarray]:
    idx: dict = {}
    for i, f in enumerate(fam):
        idx.setdefault(f, []).append(i)
    return [np.asarray(v) for v in idx.values()]


def boot_indices(groups, rng):
    pick = rng.integers(0, len(groups), len(groups))
    idx = np.concatenate([groups[k] for k in pick])
    return idx[rng.integers(0, len(idx), len(idx))]


def gap_ci(sub_lo: pd.DataFrame, sub_hi: pd.DataFrame, n=2000, seed=0):
    """Two-level bootstrap CI for r(high bin) - r(low bin)."""
    rng = np.random.default_rng(seed)
    gl, gh = family_groups(sub_lo.family.values), family_groups(sub_hi.family.values)
    yl, pl = sub_lo.pK.values, sub_lo.prediction.values
    yh, ph = sub_hi.pK.values, sub_hi.prediction.values
    out = []
    for _ in range(n):
        a = boot_indices(gh, rng)
        b = boot_indices(gl, rng)
        ra, rb = pearson(yh[a], ph[a]), pearson(yl[b], pl[b])
        if np.isfinite(ra) and np.isfinite(rb):
            out.append(ra - rb)
    return np.percentile(out, [2.5, 97.5]) if out else (np.nan, np.nan)


def paired_delta_ci(d: pd.DataFrame, ca: str, cb: str, n=2000, seed=0):
    """Two-level bootstrap CI for r_A - r_B on the complexes both models cover."""
    rng = np.random.default_rng(seed)
    groups = family_groups(d.family_id.values)
    y, a, b = d.pK.values, d[ca].values, d[cb].values
    out = []
    for _ in range(n):
        i = boot_indices(groups, rng)
        ra, rb = pearson(y[i], a[i]), pearson(y[i], b[i])
        if np.isfinite(ra) and np.isfinite(rb):
            out.append(ra - rb)
    out = np.array(out)
    return out.mean(), np.percentile(out, [2.5, 97.5]), float((out > 0).mean())


def load_preds(red: pd.DataFrame) -> dict:
    out = {}
    for name, f in MODELS.items():
        d = pd.read_csv(PRED / f).rename(columns={"id": "code"})
        out[name] = red.merge(d, on="code").dropna(subset=["prediction"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clusters", help="dir holding c<id>_cluster.tsv for each identity")
    ap.add_argument("--ids", nargs="+", default=["0.2", "0.3", "0.4", "0.5"])
    a = ap.parse_args()

    red = pd.read_parquet(RED).rename(columns={"id": "code"})
    res: dict = {}

    if a.clusters:
        print("=" * 88)
        print("1. Clustering sensitivity: G_m recomputed from the same predictions")
        print("=" * 88)
        res["clustering_sensitivity"] = {}
        for cid in a.ids:
            tsv = Path(a.clusters) / f"c{cid}_cluster.tsv"
            if not tsv.exists():
                print(f"  {cid}: {tsv} missing, skipped")
                continue
            rep = {}
            for line in open(tsv):
                r, m = line.split()
                rep[m] = r
            size = pd.Series(list(rep.values())).value_counts()
            fam = red.code.map(lambda c: rep.get(c, c))
            fsize = fam.map(lambda f: int(size.get(f, 1)))
            ann = red.assign(family=fam.values, fsize=fsize.values)
            ann["fbin"] = ann.fsize.map(family_bin)
            print(f"\n  --- {float(cid):.0%} identity: {ann.family.nunique()} families, "
                  f"singletons n={(ann.fbin == '1').sum()}, "
                  f"301+ n={(ann.fbin == '301+').sum()} ---")
            res["clustering_sensitivity"][cid] = dict(
                n_families=int(ann.family.nunique()),
                n_singleton=int((ann.fbin == "1").sum()),
                n_redundant=int((ann.fbin == "301+").sum()), models={})
            for name, f in MODELS.items():
                d = pd.read_csv(PRED / f).rename(columns={"id": "code"})
                m = ann.merge(d, on="code").dropna(subset=["prediction"])
                m = m[m.pK.between(4.5, 8.0)]
                lo, hi = m[m.fbin == "1"], m[m.fbin == "301+"]
                if len(lo) < 10 or len(hi) < 10:
                    print(f"    {name:12s} underpowered (n={len(lo)}/{len(hi)})")
                    continue
                rl, rh = pearson(lo.pK.values, lo.prediction.values), \
                    pearson(hi.pK.values, hi.prediction.values)
                ci = gap_ci(lo, hi)
                res["clustering_sensitivity"][cid]["models"][name] = dict(
                    G=float(rh - rl), r_singleton=float(rl), r_redundant=float(rh),
                    n_singleton=int(len(lo)), n_redundant=int(len(hi)),
                    ci=[float(ci[0]), float(ci[1])])
                print(f"    {name:12s} G_m={rh-rl:+.2f} [{ci[0]:+.2f},{ci[1]:+.2f}]  "
                      f"(r1={rl:+.2f} n={len(lo)}, r301={rh:+.2f} n={len(hi)})")

    print("\n" + "=" * 88)
    print("2. Novel-family ranking: paired contrasts on shared complexes")
    print("=" * 88)
    preds = load_preds(red)
    res["ranking_contrasts"] = {}
    pairs = [("RF-QSAR", "Nesso-1"), ("RF-QSAR", "Boltz-2"),
             ("ligand-kNN", "Nesso-1"), ("ligand-kNN", "Boltz-2"),
             ("Nesso-1", "Boltz-2")]
    print("  as reported, each model on its own coverage (S_f<=5, pooled):")
    for name, m in preds.items():
        s = m[m.family_size <= 5]
        print(f"    {name:12s} r={pearson(s.pK.values, s.prediction.values):+.3f}  n={len(s)}")
    print("\n  paired on shared complexes, two-level bootstrap:")
    for ca, cb in pairs:
        A = preds[ca][preds[ca].family_size <= 5][["code", "pK", "family_id", "prediction"]]
        B = preds[cb][preds[cb].family_size <= 5][["code", "prediction"]]
        d = A.merge(B, on="code", suffixes=("_a", "_b"))
        if len(d) < 20:
            print(f"    {ca} vs {cb}: n={len(d)} too few")
            continue
        ra = pearson(d.pK.values, d.prediction_a.values)
        rb = pearson(d.pK.values, d.prediction_b.values)
        delta, ci, pgt = paired_delta_ci(d, "prediction_a", "prediction_b")
        res["ranking_contrasts"][f"{ca} - {cb}"] = dict(
            n=int(len(d)), n_families=int(d.family_id.nunique()), r_a=float(ra), r_b=float(rb),
            delta=float(ra - rb), ci=[float(ci[0]), float(ci[1])], p_gt_zero=pgt)
        print(f"    {ca:11s} {ra:+.3f}  vs  {cb:11s} {rb:+.3f}   "
              f"delta={ra-rb:+.3f} [{ci[0]:+.3f},{ci[1]:+.3f}]  P(>0)={pgt:.2f}  "
              f"(n={len(d)}, {d.family_id.nunique()} families)")

    res["family_balanced"] = family_balanced(red)
    res["leave_one_family_out"] = leave_one_family_out(red)
    res["spearman_gap"] = spearman_gap(red)
    res["coverage_audit"] = coverage_audit(red)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1))
    print(f"\nwrote {OUT}")




# ---------------------------------------------------------------------------
# Family-balanced estimand (added for the reviewer's weighting objection)
# ---------------------------------------------------------------------------

def weighted_pearson(x, y, w) -> float:
    """Pearson r with per-complex weights (weights need not be normalised)."""
    w = np.asarray(w, float)
    if len(x) < 4 or w.sum() <= 0:
        return float("nan")
    w = w / w.sum()
    mx, my = (w * x).sum(), (w * y).sum()
    vx, vy = (w * (x - mx) ** 2).sum(), (w * (y - my) ** 2).sum()
    if vx <= 0 or vy <= 0:
        return float("nan")
    return float((w * (x - mx) * (y - my)).sum() / np.sqrt(vx * vy))


def family_weights(fam: np.ndarray) -> np.ndarray:
    """w_i = 1 / n_{f(i)}, so every family carries equal total mass in the stratum.

    Weights are fixed from the observed stratum, not recomputed inside a bootstrap
    replicate: recomputing them would give a family drawn twice the same total mass as
    one drawn once, which is exactly the family-level variability the bootstrap exists
    to measure.
    """
    n = pd.Series(fam).map(pd.Series(fam).value_counts())
    return (1.0 / n.values).astype(float)


def balanced_gap_ci(lo: pd.DataFrame, hi: pd.DataFrame, n=2000, seed=0):
    """Two-level bootstrap CI for the family-balanced gap."""
    rng = np.random.default_rng(seed)
    out = []
    packs = []
    for d in (hi, lo):
        packs.append((family_groups(d.family_id.values), d.pK.values,
                      d.prediction.values, family_weights(d.family_id.values)))
    for _ in range(n):
        vals = []
        for groups, y, p, w in packs:
            i = boot_indices(groups, rng)
            vals.append(weighted_pearson(p[i], y[i], w[i]))
        if all(np.isfinite(v) for v in vals):
            out.append(vals[0] - vals[1])
    return (np.percentile(out, [2.5, 97.5]) if out else (np.nan, np.nan))


def wls_cr1(X, y, w, groups):
    """Weighted least squares with CR1 cluster-robust standard errors."""
    W = np.asarray(w, float)
    XtWX_inv = np.linalg.pinv(X.T @ (X * W[:, None]))
    b = XtWX_inv @ (X.T @ (W * y))
    e = y - X @ b
    k = X.shape[1]
    meat = np.zeros((k, k))
    g = pd.Series(groups)
    for idx in g.groupby(g).indices.values():
        s = X[idx].T @ (W[idx] * e[idx])
        meat += np.outer(s, s)
    n_g = g.nunique()
    scale = (n_g / max(n_g - 1, 1)) * ((len(y) - 1) / max(len(y) - k, 1))
    return b, np.sqrt(np.diag(XtWX_inv @ (scale * meat) @ XtWX_inv)), n_g


def family_balanced(red: pd.DataFrame) -> dict:
    # family affinity variance is a corpus-level covariate, defined exactly as in
    # scripts/revision_stats.py, so the two regressions are directly comparable
    red = red.join(red.groupby("family_id").pK.var().rename("famvar"), on="family_id")
    red["famvar"] = red.famvar.fillna(0.0)
    """G_m under the complex-weighted and family-balanced estimands, plus a WLS
    version of the continuous error regression."""
    res = {"gap": {}, "regression": {}, "bin_detail": {}}
    print("\n" + "=" * 88)
    print("3. Family-balanced estimand: every family carries equal mass within a stratum")
    print("=" * 88)
    print(f"  {'model':12s} {'G_m complex-wt':>15s} {'G_m family-bal':>15s} {'95% CI (bal)':>20s}")
    for name, f in MODELS.items():
        d = pd.read_csv(PRED / f).rename(columns={"id": "code"})
        m = red.merge(d, on="code").dropna(subset=["prediction"])
        m = m[m.pK.between(4.5, 8.0)]
        m["fbin"] = m.family_size.map(family_bin)
        lo, hi = m[m.fbin == "1"], m[m.fbin == "301+"]
        if len(lo) < 10 or len(hi) < 10:
            continue
        rows = {}
        for b in BINS:
            s = m[m.fbin == b]
            if len(s) < 4:
                continue
            w = family_weights(s.family_id.values)
            rows[b] = dict(n=int(len(s)), n_families=int(s.family_id.nunique()),
                           r_complex=pearson(s.pK.values, s.prediction.values),
                           r_balanced=weighted_pearson(s.prediction.values, s.pK.values, w),
                           max_family_share=float(s.family_id.value_counts().iloc[0] / len(s)))
        gc = rows["301+"]["r_complex"] - rows["1"]["r_complex"]
        gb = rows["301+"]["r_balanced"] - rows["1"]["r_balanced"]
        ci = balanced_gap_ci(lo, hi)
        res["gap"][name] = dict(G_complex=float(gc), G_balanced=float(gb),
                                ci_balanced=[float(ci[0]), float(ci[1])])
        res["bin_detail"][name] = rows
        print(f"  {name:12s} {gc:+15.2f} {gb:+15.2f} {f'[{ci[0]:+.2f},{ci[1]:+.2f}]':>20s}")

    print("\n  per-bin r, complex-weighted vs family-balanced:")
    for name in ["Nesso-1", "RF-QSAR"]:
        if name not in res["bin_detail"]:
            continue
        print(f"    {name}")
        for b, v in res["bin_detail"][name].items():
            print(f"      {b:7s} n={v['n']:5d} fams={v['n_families']:4d} "
                  f"largest={v['max_family_share']:.0%}  "
                  f"complex {v['r_complex']:+.3f}  balanced {v['r_balanced']:+.3f}")

    print("\n  continuous regression, |error| ~ log10 S_f + covariates, weights 1/n_f:")
    print(f"    {'model':12s} {'coef (unwt)':>12s} {'t':>7s} {'coef (1/n_f)':>13s} {'t':>7s}")
    for name, f in MODELS.items():
        d = pd.read_csv(PRED / f).rename(columns={"id": "code"})
        m = red.merge(d, on="code").dropna(subset=["prediction"])
        y = m.pK.values
        pcal = np.c_[np.ones(len(m)), m.prediction.values.astype(float)]
        ae = np.abs(pcal @ np.linalg.lstsq(pcal, y, rcond=None)[0] - y)
        X = np.column_stack([
            np.ones(len(m)), np.log10(m.family_size.values), m.ligand_nn_tanimoto.values,
            np.log10(m.sequence.str.len().values), (m.year.values - 2000) / 10.0,
            m.famvar.values,
            (m.affinity_type.values == "Ki").astype(float),
            (m.affinity_type.values == "IC50").astype(float), y, y ** 2])
        w1 = np.ones(len(m))
        wf = family_weights(m.family_id.values)
        b0, se0, _ = wls_cr1(X, ae, w1, m.family_id.values)
        b1, se1, ng = wls_cr1(X, ae, wf, m.family_id.values)
        res["regression"][name] = dict(coef_unweighted=float(b0[1]),
                                       t_unweighted=float(b0[1] / se0[1]),
                                       coef_weighted=float(b1[1]),
                                       t_weighted=float(b1[1] / se1[1]),
                                       n=int(len(m)), n_families=int(ng))
        print(f"    {name:12s} {b0[1]:+12.3f} {b0[1]/se0[1]:+7.2f} "
              f"{b1[1]:+13.3f} {b1[1]/se1[1]:+7.2f}")
    return res




# ---------------------------------------------------------------------------
# Reviewer round 3: leave-one-family-out endpoint, Spearman G_m, coverage audit
# ---------------------------------------------------------------------------

def spearman(y, p) -> float:
    if len(y) < 4 or np.std(y) == 0 or np.std(p) == 0:
        return float("nan")
    return float(stats.spearmanr(p, y)[0])


def leave_one_family_out(red: pd.DataFrame) -> dict:
    """Drop each high-support family in turn and recompute the endpoint gap.

    The S_f>=301 stratum is supplied by only seven families, so the natural objection to
    the endpoint statistic is that one family carries it. This reports G_m with each of
    those families removed in turn.
    """
    out = {}
    print("\n" + "=" * 88)
    print("4. Leave-one-high-support-family-out: endpoint G_m with each 301+ family dropped")
    print("=" * 88)
    for name, f in MODELS.items():
        d = pd.read_csv(PRED / f).rename(columns={"id": "code"})
        m = red.merge(d, on="code").dropna(subset=["prediction"])
        m = m[m.pK.between(4.5, 8.0)]
        m["fbin"] = m.family_size.map(family_bin)
        lo, hi = m[m.fbin == "1"], m[m.fbin == "301+"]
        if len(lo) < 10 or len(hi) < 10:
            continue
        r_lo = pearson(lo.pK.values, lo.prediction.values)
        full = pearson(hi.pK.values, hi.prediction.values) - r_lo
        rows = []
        for fam, grp in hi.groupby("family_id"):
            rest = hi[hi.family_id != fam]
            g = pearson(rest.pK.values, rest.prediction.values) - r_lo
            rows.append(dict(family=str(fam), n_dropped=int(len(grp)),
                             share=float(len(grp) / len(hi)), G_without=float(g)))
        rows.sort(key=lambda r: -r["n_dropped"])
        gs = [r["G_without"] for r in rows]
        out[name] = dict(G_full=float(full), n_families=len(rows),
                         G_min=float(np.nanmin(gs)), G_max=float(np.nanmax(gs)), drops=rows)
        print(f"\n  {name}: full G_m={full:+.3f}, {len(rows)} families in the 301+ cell; "
              f"leave-one-out range [{np.nanmin(gs):+.3f}, {np.nanmax(gs):+.3f}]")
        for r in rows:
            print(f"    drop {r['family']:6s} (n={r['n_dropped']:4d}, {r['share']:5.1%} of cell) "
                  f"-> G_m={r['G_without']:+.3f}")
    return out


def spearman_gap(red: pd.DataFrame) -> dict:
    """G_m computed with Spearman rho instead of Pearson r, same matched window."""
    out = {}
    print("\n" + "=" * 88)
    print("5. G_m under Spearman rho (metric robustness)")
    print("=" * 88)
    print(f"  {'model':12s} {'G_m Pearson':>12s} {'G_m Spearman':>13s} {'95% CI (Spearman)':>22s}")
    for name, f in MODELS.items():
        d = pd.read_csv(PRED / f).rename(columns={"id": "code"})
        m = red.merge(d, on="code").dropna(subset=["prediction"])
        m = m[m.pK.between(4.5, 8.0)]
        m["fbin"] = m.family_size.map(family_bin)
        lo, hi = m[m.fbin == "1"], m[m.fbin == "301+"]
        if len(lo) < 10 or len(hi) < 10:
            continue
        gp = (pearson(hi.pK.values, hi.prediction.values)
              - pearson(lo.pK.values, lo.prediction.values))
        gs = (spearman(hi.pK.values, hi.prediction.values)
              - spearman(lo.pK.values, lo.prediction.values))
        rng = np.random.default_rng(0)
        packs = [(family_groups(d_.family_id.values), d_.pK.values, d_.prediction.values)
                 for d_ in (hi, lo)]
        draws = []
        for _ in range(2000):
            vals = []
            for groups, y, p in packs:
                i = boot_indices(groups, rng)
                vals.append(spearman(y[i], p[i]))
            if all(np.isfinite(v) for v in vals):
                draws.append(vals[0] - vals[1])
            ci = np.percentile(draws, [2.5, 97.5]) if draws else (np.nan, np.nan)
        out[name] = dict(G_pearson=float(gp), G_spearman=float(gs),
                         ci_spearman=[float(ci[0]), float(ci[1])])
        print(f"  {name:12s} {gp:+12.2f} {gs:+13.2f} {f'[{ci[0]:+.2f},{ci[1]:+.2f}]':>22s}")
    return out


def coverage_audit(red: pd.DataFrame) -> dict:
    """Is Boltz-2's 583-complex subset systematically different from the core set?"""
    print("\n" + "=" * 88)
    print("6. Boltz-2 coverage audit: is the 583-complex subset biased?")
    print("=" * 88)
    core = red[red.in_core] if "in_core" in red.columns else red
    b = set(pd.read_csv(PRED / MODELS["Boltz-2"]).id)
    n = set(pd.read_csv(PRED / MODELS["Nesso-1"]).id)
    core = core.assign(seq_len=core.sequence.str.len(),
                       in_boltz=core.code.isin(b), in_nesso=core.code.isin(n))
    out = {"n_core": int(len(core)), "n_boltz": int(core.in_boltz.sum()),
           "n_nesso": int(core.in_nesso.sum())}
    print(f"  core={len(core)}, Boltz-2 covers {core.in_boltz.sum()}, "
          f"Nesso-1 covers {core.in_nesso.sum()}")
    print(f"  {'variable':22s} {'Boltz covered':>14s} {'not covered':>13s} "
          f"{'p (Mann-Whitney)':>18s}")
    for col in ["seq_len", "pK", "family_size", "ligand_nn_tanimoto", "year"]:
        a, c = core[core.in_boltz][col], core[~core.in_boltz][col]
        p = float(stats.mannwhitneyu(a, c).pvalue)
        out[col] = dict(median_covered=float(a.median()), median_not=float(c.median()), p=p)
        print(f"  {col:22s} {a.median():14.1f} {c.median():13.1f} {p:18.2e}")
    bins = pd.crosstab(core.redundancy_bin, core.in_boltz).reindex(BINS)
    out["by_bin"] = json.loads(bins.to_json())
    print("\n  coverage by support bin (core subset):")
    print((bins.assign(frac=(bins[True] / (bins[True] + bins[False])).round(3))).to_string())
    return out


if __name__ == "__main__":
    main()
