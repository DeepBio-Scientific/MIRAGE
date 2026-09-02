"""Batch Boltz-2 affinity predictions over a MIRAGE target list (queued compute).

Why: the submitted Boltz-2 arm covers 583 of 18,759 complexes, which leaves the
assay-type sensitivity analysis underpowered -- the S_f>=301 / Kd cell holds only
6 complexes. This script extends coverage where it changes an inference, namely
the two extreme support bins inside the matched pK window.

    # what would run, and how many, without touching the GPU:
    python scripts/run_boltz2_affinity.py --plan

    # the run itself (needs the `boltz` conda env and a free GPU):
    conda run -n boltz python scripts/run_boltz2_affinity.py \
        --out-dir runs/boltz2_expand --assay Kd Ki --per-cell 300

    # fold the finished predictions into the shipped CSV:
    python scripts/run_boltz2_affinity.py --collect runs/boltz2_expand \
        --merge paper/predictions/boltz2_redundancy.csv

Boltz-2 reports ``affinity_pred_value`` as log10(IC50 / uM); MIRAGE stores
pAff = 6 - affinity_pred_value so that higher is stronger, matching the
convention used for the shipped predictions.
"""
from __future__ import annotations

import argparse
import glob
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RED = ROOT / "data" / "redundancy.parquet"
SHIPPED = ROOT / "paper" / "predictions" / "boltz2_redundancy.csv"

YAML = """version: 1
sequences:
  - protein:
      id: A
      sequence: {seq}
  - ligand:
      id: B
      smiles: '{smi}'
properties:
  - affinity:
      binder: B
"""


def targets(assay, per_cell, pk_window, bins_only) -> pd.DataFrame:
    df = pd.read_parquet(RED)
    df = df[df.pK.between(*pk_window) & df.affinity_type.isin(assay)]
    if bins_only:
        df = df[(df.family_size == 1) | (df.family_size >= 301)]
    df = df[~df.id.isin(set(pd.read_csv(SHIPPED).id))]
    df["cell"] = df.family_size.map(lambda s: "1" if s == 1 else "301+") + "/" + df.affinity_type
    # deterministic pick: most ligand-novel first, so the cells are not congeneric blocks
    df = df.sort_values(["cell", "ligand_nn_tanimoto", "id"])
    return df.groupby("cell", group_keys=False).head(per_cell)


def run_one(row, out_dir: Path, use_msa_server: bool) -> bool:
    work = out_dir / row.id
    if (work / "done.json").exists():
        return True
    work.mkdir(parents=True, exist_ok=True)
    yml = work / f"{row.id}.yaml"
    yml.write_text(YAML.format(seq=row.sequence, smi=row.smiles))
    cmd = ["boltz", "predict", str(yml), "--out_dir", str(work), "--override"]
    if use_msa_server:
        cmd.append("--use_msa_server")
    p = subprocess.run(cmd, capture_output=True, text=True)
    hits = glob.glob(str(work / "**" / f"affinity_{row.id}.json"), recursive=True)
    if not hits:
        (work / "failed.log").write_text((p.stdout or "") + (p.stderr or ""))
        return False
    v = json.load(open(hits[0]))["affinity_pred_value"]
    (work / "done.json").write_text(json.dumps({"id": row.id, "prediction": 6.0 - float(v)}))
    return True


def collect(out_dir: Path) -> pd.DataFrame:
    rows = [json.load(open(f)) for f in glob.glob(str(out_dir / "*" / "done.json"))]
    return pd.DataFrame(rows, columns=["id", "prediction"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="runs/boltz2_expand")
    ap.add_argument("--assay", nargs="+", default=["Kd", "Ki"])
    ap.add_argument("--per-cell", type=int, default=300)
    ap.add_argument("--pk-window", nargs=2, type=float, default=[4.5, 8.0])
    ap.add_argument("--all-bins", action="store_true",
                    help="do not restrict to the S_f=1 and S_f>=301 bins")
    ap.add_argument("--no-msa-server", action="store_true")
    ap.add_argument("--plan", action="store_true", help="print the work and exit")
    ap.add_argument("--collect", metavar="DIR", help="gather finished runs into a CSV")
    ap.add_argument("--merge", metavar="CSV", help="with --collect: union into this CSV")
    a = ap.parse_args()

    if a.collect:
        new = collect(Path(a.collect))
        print(f"collected {len(new)} predictions from {a.collect}")
        if a.merge:
            old = pd.read_csv(a.merge)
            out = pd.concat([old, new[~new.id.isin(set(old.id))]], ignore_index=True)
            out.to_csv(a.merge, index=False)
            print(f"{a.merge}: {len(old)} -> {len(out)} rows")
        else:
            print(new.to_csv(index=False))
        return

    todo = targets(a.assay, a.per_cell, a.pk_window, not a.all_bins)
    print(f"{len(todo)} complexes to run (assay={a.assay}, per_cell={a.per_cell})")
    print(todo.groupby("cell").size().to_string())
    if a.plan:
        print("\n--plan only; rerun without it inside the boltz env to execute")
        return
    if not shutil_which("boltz"):
        sys.exit("boltz not on PATH -- run inside the boltz conda env")

    out_dir = Path(a.out_dir)
    ok = 0
    for i, row in enumerate(todo.itertuples(), 1):
        ok += run_one(row, out_dir, not a.no_msa_server)
        print(f"[{i}/{len(todo)}] {row.id} ok={ok}", flush=True)
    print(f"done: {ok}/{len(todo)} succeeded -> {out_dir}")


def shutil_which(x):
    from shutil import which
    return which(x)


if __name__ == "__main__":
    main()
