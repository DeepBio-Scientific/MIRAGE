"""Load the MIRAGE evaluation sets.

Two sets ship with the benchmark:

* ``redundancy`` -- 18,759 protein-ligand complexes (PDBbind-derived) annotated with
  protein-family size, so accuracy can be stratified by how heavily the family is
  represented in the PDB. Column ``in_core`` marks a balanced 3,360-target subset for
  quick evaluation.
* ``temporal`` -- 649 compounds against one structurally novel, post-2021-09-30 target
  (OpenBind EV-A71 2A protease, CC0), with molecular-weight / clogp / Boltz-2 baselines.

Data resolves in this order: an explicit ``path``; ``$MIRAGE_DATA``; the packaged
``data/`` directory; then the Hugging Face Hub (``datasets`` extra required).
"""
from __future__ import annotations

import os
from pathlib import Path

HF_REPO = "DeepBioScientific/mirage"  # set to your namespace before pushing
_PKG_DATA = Path(__file__).resolve().parent.parent / "data"


def _resolve(name: str, path: str | None):
    fname = f"{name}.parquet"
    for base in (path, os.environ.get("MIRAGE_DATA"), _PKG_DATA):
        if base and (Path(base) / fname).exists():
            return Path(base) / fname
    return None


def load(name: str = "redundancy", *, path: str | None = None, core: bool = False):
    """Return the named set as a pandas DataFrame.

    Parameters
    ----------
    name : "redundancy" or "temporal".
    path : optional directory containing ``{name}.parquet``.
    core : for the redundancy set, return only the balanced 3,360-target subset.
    """
    import pandas as pd

    if name not in ("redundancy", "temporal"):
        raise ValueError(f"unknown set {name!r}; use 'redundancy' or 'temporal'")
    local = _resolve(name, path)
    if local is not None:
        df = pd.read_parquet(local)
    else:
        try:
            from datasets import load_dataset
        except ImportError as e:  # pragma: no cover
            raise FileNotFoundError(
                f"{name}.parquet not found locally and the 'datasets' package is not "
                f"installed. Either point path=... at the data directory or "
                f"`pip install mirage[hub]`."
            ) from e
        df = load_dataset(HF_REPO, name, split="test").to_pandas()
    if core and name == "redundancy" and "in_core" in df.columns:
        df = df[df.in_core].reset_index(drop=True)
    return df
