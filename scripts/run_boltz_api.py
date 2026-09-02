"""Score MIRAGE complexes with hosted Boltz-2.1 (boltz.bio API).

IMPORTANT -- this is a *separate arm*, not an extension of the shipped Boltz-2 column:

  * different model version: the shipped predictions come from local Boltz 2.0.x, this is
    boltz-2.1;
  * different readout: the local arm uses `affinity_pred_value` (log10 IC50 in uM,
    converted to pAff = 6 - value), while the API exposes `optimization_score`
    ("binding strength ranking score ... higher values indicate stronger predicted
    binding") and `binding_confidence` (0-1 binder/non-binder). The API does not return
    the log10(IC50) scalar at all.

Mixing the two into one column would make the Boltz arm a version-and-readout mixture, so
this writes its own CSV and records both metrics plus the model version.

    # cost first -- the API bills per complex
    python scripts/run_boltz_api.py --plan --n 583

    # validation pass: exactly the complexes that already have local predictions,
    # so optimization_score can be compared against affinity_pred_value
    python scripts/run_boltz_api.py --overlap --workers 8

    # full core subset (subsumes the overlap; skips anything already done)
    python scripts/run_boltz_api.py --core --workers 8

Requires BOLTZ_API_KEY (or BOLTZBIOAPI) in the environment; never pass the key on the
command line.
"""
from __future__ import annotations

import argparse
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RED = ROOT / "data" / "redundancy.parquet"
SHIPPED = ROOT / "paper" / "predictions" / "boltz2_redundancy.csv"
OUT_CSV = ROOT / "paper" / "predictions" / "boltz21_api_redundancy.csv"
MODEL = "boltz-2.1"
_lock = threading.Lock()


def client():
    from boltz_api import Boltz
    key = os.environ.get("BOLTZ_API_KEY") or os.environ.get("BOLTZBIOAPI")
    if not key:
        raise SystemExit("set BOLTZ_API_KEY (or BOLTZBIOAPI) in the environment")
    return Boltz(api_key=key)


def payload(row, num_samples: int) -> dict:
    return {
        "entities": [
            {"type": "protein", "value": row.sequence, "chain_ids": ["A"]},
            {"type": "ligand_smiles", "value": row.smiles, "chain_ids": ["B"]},
        ],
        "binding": {"type": "ligand_protein_binding", "binder_chain_id": "B"},
        "num_samples": num_samples,
    }


def targets(a) -> pd.DataFrame:
    red = pd.read_parquet(RED)
    if a.overlap:
        sel = red[red.id.isin(set(pd.read_csv(SHIPPED).id))]
    elif a.core:
        sel = red[red.in_core]
    else:
        sel = red[red.in_core & red.pK.between(4.5, 8.0)
                  & ((red.family_size == 1) | (red.family_size >= 301))]
    if a.n:
        sel = sel.head(a.n)
    return sel


def done_ids(out_dir: Path) -> set:
    return {p.parent.name.replace("c_", "") for p in out_dir.glob("c_*/result.json")}


def score_one(cli, row, out_dir: Path, num_samples: int) -> dict | None:
    d = out_dir / f"c_{row.id}"
    res = d / "result.json"
    if res.exists():
        return json.loads(res.read_text())
    try:
        run_dir = cli.predictions.structure_and_binding.run(
            input=payload(row, num_samples), model=MODEL, root_dir=str(out_dir),
            name=f"c_{row.id}", quiet=True)
    except Exception as e:
        (d).mkdir(parents=True, exist_ok=True)
        (d / "error.log").write_text(f"{type(e).__name__}: {e}")
        return None
    mp = Path(run_dir) / "outputs" / "files" / "prediction" / "metrics.json"
    if not mp.exists():
        return None
    m = json.loads(mp.read_text())
    bm = m.get("binding_metrics", {})
    rec = {"id": row.id, "model": MODEL,
           "optimization_score": bm.get("optimization_score"),
           "binding_confidence": bm.get("binding_confidence"),
           "iptm": m.get("best_sample", {}).get("metrics", {}).get("iptm")}
    res.write_text(json.dumps(rec))
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(ROOT / "runs" / "boltz21_api"))
    ap.add_argument("--overlap", action="store_true",
                    help="the complexes that already have local Boltz-2 predictions")
    ap.add_argument("--core", action="store_true", help="the balanced 3,360 core subset")
    ap.add_argument("--n", type=int, default=0, help="cap the number of complexes")
    ap.add_argument("--num-samples", type=int, default=1)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--plan", action="store_true", help="print scope and cost, then exit")
    a = ap.parse_args()

    sel = targets(a)
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    have = done_ids(out_dir)
    todo = sel[~sel.id.isin(have)]
    print(f"{len(sel)} selected, {len(have & set(sel.id))} already done, {len(todo)} to run")
    print(f"estimated cost at $0.05/complex: ${0.05 * len(todo):,.2f}")
    print(todo.redundancy_bin.value_counts().reindex(
        ["1", "2-5", "6-20", "21-80", "81-300", "301+"]).to_string())
    if a.plan:
        return

    cli = client()
    n_ok = 0
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(score_one, cli, r, out_dir, a.num_samples): r.id
                for r in todo.itertuples()}
        for k, fut in enumerate(as_completed(futs), 1):
            rec = fut.result()
            n_ok += rec is not None
            if k % 10 == 0 or k == len(futs):
                print(f"[{k}/{len(futs)}] ok={n_ok}", flush=True)

    recs = [json.loads(p.read_text()) for p in out_dir.glob("c_*/result.json")]
    df = pd.DataFrame(recs)
    if not df.empty:
        # `prediction` is the column the harness scores; higher = stronger binding
        df["prediction"] = df.optimization_score
        df.to_csv(OUT_CSV, index=False)
        print(f"wrote {OUT_CSV} ({len(df)} rows)")


if __name__ == "__main__":
    main()
