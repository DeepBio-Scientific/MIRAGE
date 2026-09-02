"""Resumable ESMFold2 (Forge) completion pass for the MIRAGE confidence-proxy arm.

Forge caps at ~100 folds/day, so completing the 600-target balanced set takes several
daily runs. This script SKIPS already-folded complexes and stops gracefully at the daily
credit cap, so run it once per day (e.g. via crontab) until it reports 600/600:

    BIOHUB_API_KEY=... python scripts/esmfold2_forge_complete.py --targets codes.json

The key is read from the environment and never printed; pass it in the environment rather
than on the command line. Sequences and SMILES come from data/redundancy.parquet, and
ipTM values are appended to runs/esmfold2_iptm.json after every fold.

`--targets` is a JSON list of PDB codes. The released numbers used the same 600 codes as
the Chai-1 arm; without the file this falls back to a deterministic balanced draw of 100
per redundancy bin from the core set, which is the same construction but not the same
draw. Note: at n=281 the ESMFold2 ipTM curve is already flat (no family-support
dependence, t=-0.7); completing to 600 is for thoroughness and is not expected to change
that conclusion.
"""
import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RED = ROOT / "data" / "redundancy.parquet"
PER_BIN = 100


def balanced_codes(red, per_bin=PER_BIN, seed=0):
    """Deterministic balanced draw across redundancy bins, from the core set."""
    core = red[red.in_core]
    picks = pd.concat([g.sample(min(per_bin, len(g)), random_state=seed)
                       for _, g in core.groupby("redundancy_bin")])
    return sorted(picks.id)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--targets", help="JSON list of PDB codes (default: balanced draw)")
    ap.add_argument("--out", help="output JSON (default runs/esmfold2_iptm.json)")
    ap.add_argument("--max-consecutive-failures", type=int, default=5,
                    help="stop after this many in a row -- the daily cap looks like this")
    a = ap.parse_args()

    tok = os.environ.get("BIOHUB_API_KEY")
    if not tok:
        raise SystemExit("set BIOHUB_API_KEY in the environment")

    from esm.sdk import esmfold2_client
    from esm.sdk.api import FoldingConfig
    from esm.utils.structure.input_builder import (
        LigandInput,
        ProteinInput,
        StructurePredictionInput,
    )

    red = pd.read_parquet(RED)
    seqs = dict(zip(red.id, red.sequence))
    smiles = dict(zip(red.id, red.smiles))
    codes = json.loads(Path(a.targets).read_text()) if a.targets else balanced_codes(red)
    target_n = len(codes)

    out_path = Path(a.out) if a.out else ROOT / "runs" / "esmfold2_iptm.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out = json.loads(out_path.read_text()) if out_path.exists() else {}
    todo = [c for c in codes if c not in out and c in seqs and c in smiles]
    print(f"have {len(out)}, remaining to complete {target_n}: {len(todo)}", flush=True)

    client = esmfold2_client(token=tok)
    cfg = FoldingConfig(num_loops=3, num_sampling_steps=30, include_pae=True)
    done = 0
    capped = False
    consec = 0
    t0 = time.time()
    for i, code in enumerate(todo):
        try:
            inp = StructurePredictionInput(sequences=[
                ProteinInput(id="A", sequence=seqs[code][:1000]),
                LigandInput(id="B", ccd=None, smiles=smiles[code])])
            r = client.fold_all_atom(inp, config=cfg)
            pci = getattr(r, "pair_chains_iptm", None)
            val = None
            if isinstance(pci, dict):
                try:
                    val = float(pci["A"]["B"])
                except Exception:
                    val = None
            if val is None:
                val = float(r.iptm)
            out[code] = val
            done += 1
            consec = 0
            out_path.write_text(json.dumps(out))
        except Exception:
            consec += 1
            if consec >= a.max_consecutive_failures:
                print(f"CAP/ERROR: {consec} consecutive failures after +{done} "
                      f"(total {len(out)}/{target_n}); stopping", flush=True)
                capped = True
                break
        time.sleep(3.2)
        if (i + 1) % 10 == 0:
            print(f"{i + 1}/{len(todo)} +{done} {time.time() - t0:.0f}s", flush=True)

    out_path.write_text(json.dumps(out))
    print(f"PASS DONE: +{done} new, total {len(out)}/{target_n}, capped={capped}", flush=True)
    print("COMPLETE_DONE" if len(out) >= target_n - 5
          else ("CAPPED" if capped else "PASS_END"), flush=True)


if __name__ == "__main__":
    main()
