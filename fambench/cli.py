"""Command-line interface: score a predictions CSV without writing any Python.

    fambench score redundancy preds.csv
    fambench score temporal preds.csv --pred-col affinity --lower-is-stronger
    fambench baselines           # run the shipped MW / clogp baselines
    fambench info                # dataset sizes and columns
"""
from __future__ import annotations

import argparse
import sys


def _cmd_score(a):
    from .runner import score_csv

    rep = score_csv(a.dataset, a.csv, id_col=a.id_col, pred_col=a.pred_col,
                    higher_is_stronger=not a.lower_is_stronger, path=a.data)
    print(rep.summary())


def _cmd_baselines(a):
    from .runner import baseline_model, run_redundancy, run_temporal

    for name in ("molecular_weight", "clogp"):
        m = baseline_model(name)
        print(f"\n### baseline: {name}")
        print(run_redundancy(m, core=not a.full, path=a.data, progress=False).summary())
        print(run_temporal(m, path=a.data, progress=False).summary())


def _cmd_info(a):
    from .data import load

    for name in ("redundancy", "temporal"):
        df = load(name, path=a.data)
        print(f"\n{name}: {len(df)} rows")
        print("  columns:", ", ".join(df.columns))
        if name == "redundancy":
            print("  families:", df.family_id.nunique(),
                  "| core subset:", int(df.in_core.sum()))
            print("  redundancy_bin counts:",
                  df.redundancy_bin.value_counts().to_dict())


def main(argv=None):
    p = argparse.ArgumentParser(prog="fambench", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", help="directory containing redundancy.parquet / temporal.parquet")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("score", help="score a predictions CSV")
    s.add_argument("dataset", choices=["redundancy", "temporal"])
    s.add_argument("csv", help="CSV with columns: id, prediction")
    s.add_argument("--id-col", default="id")
    s.add_argument("--pred-col", default="prediction")
    s.add_argument("--lower-is-stronger", action="store_true",
                   help="set if your score is like log10(IC50) (lower = stronger)")
    s.set_defaults(func=_cmd_score)

    b = sub.add_parser("baselines", help="run the shipped MW / clogp baselines")
    b.add_argument("--full", action="store_true", help="use full redundancy set, not core")
    b.set_defaults(func=_cmd_baselines)

    i = sub.add_parser("info", help="show dataset sizes and columns")
    i.set_defaults(func=_cmd_info)

    a = p.parse_args(argv)
    a.func(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
