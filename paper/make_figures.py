"""Regenerate the paper figures from results/*.json. Requires matplotlib."""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.grid": True,
                     "grid.alpha": 0.3, "figure.dpi": 150})
HERE = os.path.dirname(os.path.abspath(__file__))
red = json.load(open(f"{HERE}/results/redundancy_results.json"))
tmp = json.load(open(f"{HERE}/results/temporal_results.json"))
os.makedirs(f"{HERE}/figures", exist_ok=True)

BINS = ["1", "2-5", "6-20", "21-80", "81-300", "301+"]
xlab = ["1\n(singleton)", "2–5", "6–20", "21–80", "81–300", "301+"]
COL = {"Nesso-1": "#1f77b4", "Boltz-2": "#d62728", "RF-QSAR": "#2ca02c",
       "ligand-kNN": "#9467bd", "mol.weight": "#7f7f7f", "clogp": "#bcbd22"}
MARK = {"Nesso-1": "o", "Boltz-2": "s", "RF-QSAR": "^", "ligand-kNN": "D",
        "mol.weight": "v", "clogp": "P"}

# Fig 1: dose-response
fig, ax = plt.subplots(figsize=(6.2, 4.2))
for name in ["Nesso-1", "Boltz-2", "RF-QSAR", "ligand-kNN", "mol.weight"]:
    m = red[name]["matched"]
    xs = [i for i, b in enumerate(BINS) if b in m]
    ys = [m[b]["pearson"] for b in BINS if b in m]
    lo = [m[b]["pearson"] - m[b]["ci"][0] for b in BINS if b in m]
    hi = [m[b]["ci"][1] - m[b]["pearson"] for b in BINS if b in m]
    ls = "--" if name == "mol.weight" else "-"
    ax.errorbar(xs, ys, yerr=[lo, hi], marker=MARK[name], color=COL[name],
                label=name, ls=ls, capsize=2, lw=1.6, ms=5)
ax.axhline(0, color="k", lw=0.6)
ax.set_xticks(range(6))
ax.set_xticklabels(xlab)
ax.set_xlabel("protein family size (MMseqs2 30% cluster)")
ax.set_ylabel("Pearson $r$ (matched-p$K$ window)")
ax.set_title("Affinity accuracy vs. protein-family redundancy")
ax.legend(fontsize=8, ncol=2, loc="upper left")
fig.tight_layout()
fig.savefig(f"{HERE}/figures/fig1_dose_response.pdf")
plt.close(fig)

# Fig 2: novel vs redundant
fig, ax = plt.subplots(figsize=(6.2, 3.8))
names = ["Nesso-1", "Boltz-2", "RF-QSAR", "ligand-kNN", "mol.weight", "clogp"]
x = np.arange(len(names))
w = 0.38
ax.bar(x - w / 2, [red[n]["novel"] for n in names], w,
       label="novel families (size 1–5)", color="#4c72b0")
ax.bar(x + w / 2, [red[n]["redundant"] for n in names], w,
       label="redundant families (size $\\geq$81)", color="#dd8452")
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=20, ha="right")
ax.set_ylabel("Pearson $r$")
ax.axhline(0, color="k", lw=0.6)
ax.set_title("Novel vs. redundant families: the ranking inverts")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(f"{HERE}/figures/fig2_novel_vs_redundant.pdf")
plt.close(fig)

# Fig 3: leakage slope
fig, ax = plt.subplots(figsize=(5.6, 3.6))
cols = ["#c44e52" if red[n]["slope_t"] < -2 else "#8c8c8c" for n in names]
ax.barh(names[::-1], [red[n]["slope"] for n in names][::-1], color=cols[::-1])
for i, n in enumerate(names[::-1]):
    s = red[n]["slope"]
    ax.text(s - 0.005 if s < 0 else s + 0.005, i, f"t={red[n]['slope_t']:+.1f}",
            va="center", ha="right" if s < 0 else "left", fontsize=7.5)
ax.axvline(0, color="k", lw=0.6)
ax.set_xlabel("leakage slope: $\\partial$|error| / $\\partial\\log_{10}$(family size)")
ax.set_title("Only co-folding models are redundancy-dependent")
fig.tight_layout()
fig.savefig(f"{HERE}/figures/fig3_leakage_slope.pdf")
plt.close(fig)

# Fig 4: temporal
fig, ax = plt.subplots(figsize=(6.2, 3.8))
order = ["Nesso-1", "mol.weight", "gnina", "Boltz-2", "smina", "AEV-PLIG",
         "clogp", "AQ-Affinity"]
cols = ["#4c72b0" if n not in ("mol.weight", "clogp") else "#7f7f7f" for n in order]
ax.bar(range(len(order)), [tmp[n]["spearman"] for n in order], color=cols)
mw = tmp["mol.weight"]["spearman"]
ax.axhline(mw, color="#c44e52", ls="--", lw=1.4, label=f"molecular weight ({mw:.2f})")
ax.set_xticks(range(len(order)))
ax.set_xticklabels(order, rotation=25, ha="right")
ax.set_ylabel("Spearman $\\rho$")
ax.set_title("Novel target (OpenBind EV-A71 2A protease)")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(f"{HERE}/figures/fig4_temporal.pdf")
plt.close(fig)
print("figures written to", f"{HERE}/figures")
