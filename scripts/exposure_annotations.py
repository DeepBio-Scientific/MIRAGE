"""Decompose training exposure: exact-target repetition vs. family support.

S_f counts structures in a 30% identity cluster, which conflates two very different
things: the same protein appearing again with a different ligand, and a remote homolog
appearing. This script annotates every complex with how many *other* corpus complexes
share its protein at three stringencies, so the two can be separated in analysis:

    n_exact_sib  identical extracted sequence
    n_sib_99     same MMseqs2 cluster at 99% identity / 90% coverage
    n_sib_95     same MMseqs2 cluster at 95% identity / 90% coverage

Writes data/exposure_annotations.parquet (id + the three counts). Requires mmseqs2 on
PATH; the output is small and versioned so the analysis reproduces without rerunning it.

    python scripts/exposure_annotations.py
"""
from __future__ import annotations

import argparse
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RED = ROOT / "data" / "redundancy.parquet"
OUT = ROOT / "data" / "exposure_annotations.parquet"


def cluster_siblings(df: pd.DataFrame, min_seq_id: float, workdir: Path) -> pd.Series:
    """Number of other complexes in the same identity cluster."""
    fa = workdir / f"in_{min_seq_id}.fasta"
    with open(fa, "w") as f:
        for i, s in zip(df.id, df.sequence):
            f.write(f">{i}\n{s}\n")
    pref = workdir / f"c{min_seq_id}"
    subprocess.run(["mmseqs", "easy-cluster", str(fa), str(pref),
                    str(workdir / f"tmp{min_seq_id}"), "--min-seq-id", str(min_seq_id),
                    "-c", "0.9", "--cov-mode", "0"],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    rep = {}
    for line in open(f"{pref}_cluster.tsv"):
        r, m = line.split()
        rep[m] = r
    size = Counter(rep.values())
    return df.id.map(lambda i: size.get(rep.get(i, i), 1) - 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--workdir", default=None)
    a = ap.parse_args()

    df = pd.read_parquet(RED)[["id", "sequence", "family_size"]]
    tmp = Path(a.workdir) if a.workdir else Path(tempfile.mkdtemp(prefix="mirage_expo_"))
    tmp.mkdir(parents=True, exist_ok=True)

    n_exact = df.sequence.map(df.groupby("sequence").size()) - 1
    out = pd.DataFrame({"id": df.id, "n_exact_sib": n_exact.values})
    for cut, col in [(0.99, "n_sib_99"), (0.95, "n_sib_95")]:
        out[col] = cluster_siblings(df, cut, tmp).values
        print(f"{col}: {(out[col] == 0).sum()} complexes with no sibling at {cut:.0%} identity")
    out.to_parquet(a.out, index=False)
    print(f"wrote {a.out} ({len(out)} rows)")


if __name__ == "__main__":
    main()
