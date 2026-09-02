"""Is protein-family support S_f just a proxy for available evolutionary signal?

The pose arm shows that supplying a ColabFold MSA rescues novel-family targets. That
raises a fair objection: family novelty here is defined by PDBbind *structural* support,
while an MSA is retrieved from far larger *sequence* databases, so a target can be
structurally novel and still have a deep sequence family. If MSA depth were simply a
restatement of S_f, the MSA result would say nothing new.

This script measures that directly. For each target it fetches a ColabFold MSA, computes
depth, Neff (sequence-weighted effective count at 80% identity), Neff per residue and the
nearest-homolog identity, then reports how those relate to log10 S_f -- and, if per-complex
pose outcomes are supplied, whether MSA depth or S_f better explains novel-family success.

    python scripts/msa_depth.py --targets data/pose_targets.json      # 100 pose targets
    python scripts/msa_depth.py --targets data/pose_targets.json \
        --pose-results /path/to/pose_smina.json

MSAs are fetched from the public ColabFold MMseqs2 server, cached under
runs/msa_cache/<id>.a3m, and re-used on later runs. Be considerate with the service:
requests are serialised with a delay, and a cached target is never re-fetched.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
RED = ROOT / "data" / "redundancy.parquet"
CACHE = ROOT / "runs" / "msa_cache"
API = "https://api.colabfold.com"


def fetch_a3m(seq: str, cid: str, poll: float = 5.0, timeout: float = 1800) -> str | None:
    """Submit one sequence to the ColabFold MMseqs2 server and return the a3m."""
    import requests

    CACHE.mkdir(parents=True, exist_ok=True)
    out = CACHE / f"{cid}.a3m"
    if out.exists():
        return out.read_text()
    q = f">{cid}\n{seq}\n"
    r = requests.post(f"{API}/ticket/msa", data={"q": q, "mode": "env"}, timeout=60)
    r.raise_for_status()
    tid = r.json()["id"]
    t0 = time.time()
    while time.time() - t0 < timeout:
        s = requests.get(f"{API}/ticket/{tid}", timeout=60).json()
        if s.get("status") == "COMPLETE":
            break
        if s.get("status") in {"ERROR", "MAINTENANCE", "RATELIMIT"}:
            print(f"  {cid}: server status {s.get('status')}")
            return None
        time.sleep(poll)
    else:
        return None
    import io
    import tarfile

    res = requests.get(f"{API}/result/download/{tid}", timeout=300)
    res.raise_for_status()
    with tarfile.open(fileobj=io.BytesIO(res.content), mode="r:gz") as tf:
        for m in tf.getmembers():
            if m.name.endswith(".a3m"):
                a3m = tf.extractfile(m).read().decode()
                out.write_text(a3m)
                return a3m
    return None


def parse_a3m(a3m: str) -> list[str]:
    """Aligned sequences (insertions stripped), query first."""
    seqs, cur = [], []
    for line in a3m.splitlines():
        if line.startswith(">"):
            if cur:
                seqs.append("".join(cur))
                cur = []
        elif line.strip():
            cur.append("".join(c for c in line.strip() if not c.islower()))
    if cur:
        seqs.append("".join(cur))
    return [s for s in seqs if s]


def msa_stats(seqs: list[str], id_cut: float = 0.8) -> dict:
    """Depth, Neff (80% id sequence weighting), Neff/L, nearest-homolog identity."""
    if not seqs:
        return {}
    L = len(seqs[0])
    keep = [s for s in seqs if len(s) == L]
    A = np.array([[ord(c) for c in s] for s in keep], dtype=np.int16)
    n = len(A)
    # pairwise identity in blocks; Neff = sum of 1/(cluster occupancy)
    counts = np.ones(n)
    for i in range(0, n, 512):
        blk = A[i:i + 512]
        same = (blk[:, None, :] == A[None, :, :]).sum(2) / max(L, 1)
        counts[i:i + blk.shape[0]] = (same >= id_cut).sum(1)
    neff = float((1.0 / counts).sum())
    nearest = float(((A[1:] == A[0]).sum(1) / max(L, 1)).max()) if n > 1 else 0.0
    return dict(depth=n, neff=neff, neff_per_res=neff / max(L, 1),
                nearest_homolog_id=nearest, length=L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default=str(ROOT / "data" / "pose_targets.json"))
    ap.add_argument("--pose-results", help="JSON of {id: {rmsd, bin}} to use as the outcome")
    ap.add_argument("--out", default=str(ROOT / "paper" / "results" / "msa_depth.json"))
    ap.add_argument("--sleep", type=float, default=2.0, help="delay between submissions")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    tj = json.load(open(a.targets))
    ids = [i for v in tj.values() for i in v] if isinstance(tj, dict) else list(tj)
    if a.limit:
        ids = ids[:a.limit]
    red = pd.read_parquet(RED).set_index("id")
    ids = [i for i in ids if i in red.index]
    print(f"{len(ids)} targets; cache {CACHE}")

    rows = []
    for k, cid in enumerate(ids, 1):
        cached = (CACHE / f"{cid}.a3m").exists()
        try:
            a3m = fetch_a3m(red.loc[cid, "sequence"], cid)
        except Exception as e:  # network/service problems should not lose prior work
            print(f"[{k}/{len(ids)}] {cid}: {type(e).__name__}: {e}")
            continue
        if not a3m:
            print(f"[{k}/{len(ids)}] {cid}: no MSA returned")
            continue
        st = msa_stats(parse_a3m(a3m))
        st.update(id=cid, family_size=int(red.loc[cid, "family_size"]))
        rows.append(st)
        print(f"[{k}/{len(ids)}] {cid}: depth={st['depth']} neff={st['neff']:.0f} "
              f"S_f={st['family_size']}")
        if not cached:
            time.sleep(a.sleep)

    df = pd.DataFrame(rows)
    if df.empty:
        print("no MSAs retrieved")
        return
    df["log_Sf"] = np.log10(df.family_size)
    df["log_neff"] = np.log10(df.neff.clip(lower=1))

    print("\n=== Does S_f proxy evolutionary depth? ===")
    out = {"n": int(len(df))}
    for col in ["log_neff", "nearest_homolog_id", "neff_per_res"]:
        r, p = stats.pearsonr(df[col], df.log_Sf)
        sr, sp = stats.spearmanr(df[col], df.log_Sf)
        out[col] = dict(pearson_r=float(r), pearson_p=float(p),
                        spearman_r=float(sr), spearman_p=float(sp))
        print(f"  {col:20s} vs log10 S_f : Pearson {r:+.3f} (p={p:.3g}), "
              f"Spearman {sr:+.3f} (p={sp:.3g})")
    for b, g in df.groupby(df.family_size.map(lambda s: "novel" if s <= 5 else
                                              "redundant" if s >= 301 else "mid")):
        print(f"  {b:10s} n={len(g):3d}  median depth {g.depth.median():8.0f}  "
              f"median Neff {g.neff.median():8.0f}")
        out[f"median_{b}"] = dict(depth=float(g.depth.median()), neff=float(g.neff.median()))

    if a.pose_results:
        pose = json.load(open(a.pose_results))
        df["success"] = df.id.map(lambda i: float(pose[i]["rmsd"] < 2.0)
                                  if i in pose and pose[i].get("rmsd") is not None else np.nan)
        d = df.dropna(subset=["success"])
        if len(d) > 10:
            X = np.column_stack([np.ones(len(d)), d.log_Sf, d.log_neff])
            b, *_ = np.linalg.lstsq(X, d.success.values, rcond=None)
            e = d.success.values - X @ b
            se = np.sqrt(np.diag(np.linalg.pinv(X.T @ X)) * (e @ e) / max(len(d) - 3, 1))
            print(f"\n  success ~ log10 S_f + log10 Neff  (n={len(d)})")
            for nm, bb, ss in zip(["const", "log10_Sf", "log10_Neff"], b, se):
                print(f"    {nm:12s} {bb:+.4f} (t={bb/ss:+.2f})")
            terms = ["const", "log10_Sf", "log10_Neff"]
            out["success_model"] = {nm: dict(coef=float(bb), t=float(bb / ss))
                                    for nm, bb, ss in zip(terms, b, se)}

    Path(a.out).write_text(json.dumps({"targets": rows, "summary": out}, indent=1))
    df.to_csv(Path(a.out).with_suffix(".csv"), index=False)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
