"""Core evaluation: family-stratified affinity accuracy with confound guards.

The central metric is not a single correlation but how correlation *scales with
protein-family redundancy*. A model that only works on families heavily represented
in the PDB is memorising, not generalising. See the README for the empirical
motivation (Nesso-1 and Boltz-2 both collapse to ~0 on singleton families).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats

REDUNDANCY_BINS = ["1", "2-5", "6-20", "21-80", "81-300", "301+"]


def _bootstrap_ci(x, y, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    m = len(x)
    for _ in range(n):
        idx = rng.integers(0, m, m)
        if np.std(y[idx]) > 0 and np.std(x[idx]) > 0:
            vals.append(stats.pearsonr(x[idx], y[idx]).statistic)
    if not vals:
        return (float("nan"), float("nan"))
    return tuple(np.percentile(vals, [2.5, 97.5]))


@dataclass
class BinResult:
    bin: str
    n: int
    pearson: float
    spearman: float
    ci_low: float
    ci_high: float
    label_sd: float
    mae: float


@dataclass
class RedundancyReport:
    n: int
    overall_pearson: float
    overall_spearman: float
    novel_family_pearson: float  # headline: accuracy on singleton+small families
    redundant_family_pearson: float
    leakage_slope: float  # OLS coef of abs-error on log10(family_size)
    leakage_slope_t: float  # cluster-robust (by family) if family_id was given
    leakage_slope_t_ols: float  # unclustered, for comparison only
    n_families: int
    bins: list[BinResult] = field(default_factory=list)
    matched_bins: list[BinResult] = field(default_factory=list)

    def summary(self) -> str:
        clustering = (f"{self.n_families} family clusters" if self.n_families
                      else "unclustered")
        lines = [
            f"MIRAGE redundancy report  (n={self.n})",
            f"  overall            Pearson {self.overall_pearson:+.3f}  "
            f"Spearman {self.overall_spearman:+.3f}",
            f"  NOVEL families     Pearson {self.novel_family_pearson:+.3f}   "
            f"<- the number that matters for a new target",
            f"  redundant families Pearson {self.redundant_family_pearson:+.3f}",
            f"  leakage slope      {self.leakage_slope:+.3f} "
            f"(t={self.leakage_slope_t:+.1f}, {clustering})"
            f"  [abs-error vs log10 family size; <0 = accuracy bought by redundancy]",
            "",
            "  family-size dose-response (matched-pK, label spread held constant):",
            f"    {'bin':>8s} {'n':>4s} {'Pearson':>8s} {'[95% CI]':>16s}",
        ]
        for b in self.matched_bins:
            lines.append(
                f"    {b.bin:>8s} {b.n:>4d} {b.pearson:+8.3f} "
                f"[{b.ci_low:+.2f},{b.ci_high:+.2f}]"
            )
        return "\n".join(lines)


def _family_bin(size: int) -> str:
    if size == 1:
        return "1"
    if size <= 5:
        return "2-5"
    if size <= 20:
        return "6-20"
    if size <= 80:
        return "21-80"
    if size <= 300:
        return "81-300"
    return "301+"


def evaluate_redundancy(
    labels,
    predictions,
    family_size,
    *,
    family_id=None,
    affinity_type=None,
    matched_pk_window=(4.5, 8.0),
    higher_is_stronger=True,
) -> RedundancyReport:
    """Family-stratified evaluation.

    Parameters
    ----------
    labels : array of experimental pK (higher = stronger binding).
    predictions : array of model scores. If ``higher_is_stronger`` is False (e.g.
        log10(IC50), lower = stronger) the sign is flipped internally so a positive
        correlation always means "good".
    family_size : array of protein-family sizes (redundancy). Provided by the
        benchmark dataset; do not compute it from your own predictions.
    family_id : optional array of family identifiers. Complexes within a family are
        not independent, so when this is supplied the family-support slope is tested
        with CR1 standard errors clustered on family. Without it the slope is tested
        with ordinary OLS errors, which overstate significance.
    affinity_type : optional array of {'Kd','Ki','IC50'} for the confound regression.
    matched_pk_window : restrict the per-bin correlations to this pK range so that
        label spread (which attenuates correlation) is comparable across bins.

    Returns a :class:`RedundancyReport`.
    """
    y = np.asarray(labels, float)
    p = np.asarray(predictions, float)
    if not higher_is_stronger:
        p = -p
    fs = np.asarray(family_size, float)
    ok = np.isfinite(y) & np.isfinite(p) & np.isfinite(fs)
    y, p, fs = y[ok], p[ok], fs[ok]
    at = np.asarray(affinity_type)[ok] if affinity_type is not None else None
    fid = np.asarray(family_id)[ok] if family_id is not None else None
    abserr = np.abs(p - (p.mean() + (y - y.mean())))  # placeholder, replaced below
    # calibrate predictions to label scale for an interpretable abs-error
    b1 = np.c_[np.ones_like(p), p]
    beta = np.linalg.lstsq(b1, y, rcond=None)[0]
    p_cal = b1 @ beta
    abserr = np.abs(p_cal - y)

    def bin_result(mask, do_ci=True):
        yy, pp, ae = y[mask], p[mask], abserr[mask]
        if len(yy) < 3 or np.std(yy) == 0 or np.std(pp) == 0:
            return None
        pr = stats.pearsonr(pp, yy).statistic
        sp = stats.spearmanr(pp, yy).statistic
        ci = _bootstrap_ci(pp, yy) if do_ci else (float("nan"), float("nan"))
        return BinResult(bin="", n=len(yy), pearson=pr, spearman=sp,
                         ci_low=ci[0], ci_high=ci[1], label_sd=float(yy.std()),
                         mae=float(ae.mean()))

    fbin = np.array([_family_bin(int(s)) for s in fs])
    bins, matched = [], []
    lo, hi = matched_pk_window
    for b in REDUNDANCY_BINS:
        r = bin_result(fbin == b)
        if r:
            r.bin = b
            bins.append(r)
        rm = bin_result((fbin == b) & (y >= lo) & (y <= hi))
        if rm:
            rm.bin = b
            matched.append(rm)

    novel = bin_result(np.isin(fbin, ["1", "2-5"]), do_ci=False)
    redun = bin_result(np.isin(fbin, ["81-300", "301+"]), do_ci=False)

    # leakage slope: abs-error ~ log10(family_size) [+ pK + pK^2 + atype]
    cols = [np.ones_like(fs), np.log10(fs), y, y ** 2]
    names = ["int", "log10_famsize", "pK", "pK2"]
    if at is not None:
        cols += [(at == "Ki").astype(float), (at == "IC50").astype(float)]
        names += ["is_Ki", "is_IC50"]
    X = np.column_stack(cols)
    XtX_inv = np.linalg.pinv(X.T @ X)
    coef = XtX_inv @ X.T @ abserr
    resid = abserr - X @ coef
    n_obs, n_par = X.shape
    dof = max(n_obs - n_par, 1)
    se_ols = np.sqrt(np.diag(XtX_inv) * (resid @ resid) / dof)
    se = se_ols
    n_fam = 0
    if fid is not None:
        # CR1 cluster-robust covariance: complexes in one family are dependent.
        meat = np.zeros((n_par, n_par))
        order = np.argsort(fid, kind="stable")
        bounds = np.flatnonzero(np.r_[True, fid[order][1:] != fid[order][:-1]])
        for a, b in zip(bounds, np.r_[bounds[1:], len(order)]):
            g = order[a:b]
            sc = X[g].T @ resid[g]
            meat += np.outer(sc, sc)
        n_fam = len(bounds)
        scale = (n_fam / max(n_fam - 1, 1)) * ((n_obs - 1) / dof)
        se = np.sqrt(np.diag(XtX_inv @ (scale * meat) @ XtX_inv))
    j = names.index("log10_famsize")

    return RedundancyReport(
        n=len(y),
        overall_pearson=float(stats.pearsonr(p, y).statistic),
        overall_spearman=float(stats.spearmanr(p, y).statistic),
        novel_family_pearson=novel.pearson if novel else float("nan"),
        redundant_family_pearson=redun.pearson if redun else float("nan"),
        leakage_slope=float(coef[j]),
        leakage_slope_t=float(coef[j] / se[j]) if se[j] else float("nan"),
        leakage_slope_t_ols=float(coef[j] / se_ols[j]) if se_ols[j] else float("nan"),
        n_families=int(n_fam),
        bins=bins,
        matched_bins=matched,
    )


@dataclass
class TemporalReport:
    n: int
    spearman: float
    pearson: float
    baseline_spearman: dict
    residual_vs_mw_spearman: float  # signal beyond molecular weight

    def summary(self) -> str:
        lines = [f"MIRAGE temporal report  (novel target, n={self.n})",
                 f"  your model         Spearman {self.spearman:+.3f}  "
                 f"Pearson {self.pearson:+.3f}"]
        for k, v in sorted(self.baseline_spearman.items(), key=lambda kv: -kv[1]):
            lines.append(f"  baseline {k:14s} Spearman {v:+.3f}")
        lines.append(f"  signal beyond MW   Spearman {self.residual_vs_mw_spearman:+.3f}  "
                     f"(residualised on molecular weight)")
        return "\n".join(lines)


def evaluate_temporal(
    labels, predictions, *, baselines=None, higher_is_stronger=True
) -> TemporalReport:
    """Within-target ranking on a genuinely novel (post-cutoff) target.

    ``baselines`` is an optional dict name -> array (e.g. {'molecular_weight': mw}).
    The key ``molecular_weight`` (if present) is also used for the residual test.
    """
    y = np.asarray(labels, float)
    p = np.asarray(predictions, float)
    if not higher_is_stronger:
        p = -p
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = y[ok], p[ok]
    bl_sp = {}
    resid = float("nan")
    if baselines:
        for name, arr in baselines.items():
            a = np.asarray(arr, float)[ok]
            m = np.isfinite(a)
            if m.sum() > 3:
                bl_sp[name] = float(stats.spearmanr(a[m], y[m]).statistic)
        if "molecular_weight" in baselines:
            mw = np.asarray(baselines["molecular_weight"], float)[ok]
            m = np.isfinite(mw)
            if m.sum() > 3:
                def _res(a, b):
                    b1 = np.c_[np.ones_like(b), b]
                    return a - b1 @ np.linalg.lstsq(b1, a, rcond=None)[0]
                resid = float(stats.spearmanr(_res(p[m], mw[m]), _res(y[m], mw[m])).statistic)
    return TemporalReport(
        n=len(y),
        spearman=float(stats.spearmanr(p, y).statistic),
        pearson=float(stats.pearsonr(p, y).statistic),
        baseline_spearman=bl_sp,
        residual_vs_mw_spearman=resid,
    )
