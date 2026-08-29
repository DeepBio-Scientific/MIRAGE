"""Score co-folding *structure-confidence* models on MIRAGE.

AlphaFold3, Chai-1, and ESMFold2 predict a bound complex and a confidence, not an
affinity. Their protein-ligand interface confidence (ipTM) is nonetheless routinely used
as a rough binding-likelihood signal. MIRAGE can score it -- clearly labeled as a
proxy, not an affinity prediction -- to test whether structural confidence alone carries
the family-redundancy leakage.

This module provides parsers that turn each model's native output into the
``id,prediction`` form the harness expects (higher = stronger). Predictions are the
protein-ligand interface ipTM.

Availability, verified 2026-07:
  * Chai-1     -- runs locally (`pip install chai_lab`); ``parse_chai_iptm`` below.
  * ESMFold2   -- NO released local inference (weights on HF but no model code;
                  served only via the Biohub Forge API). Provide a Forge token and use
                  ``forge_esmfold2_iptm``.
  * AlphaFold3 -- code open (github.com/google-deepmind/alphafold3) but weights are
                  request-only from Google DeepMind. Once obtained, parse its
                  ``*_summary_confidences.json`` (``chain_pair_iptm``) with
                  ``parse_af3_iptm``.
"""
from __future__ import annotations

import glob
import json
import os


def parse_chai_iptm(out_root: str, protein_chain: int = 0, ligand_chain: int = 1) -> dict:
    """Map complex id -> protein-ligand interface ipTM from a Chai-1 output tree.

    Expects ``out_root/<id>/scores.model_idx_*.npz`` (the default `chai-lab fold` layout),
    averaging the interface ipTM over the sampled models.
    """
    import numpy as np

    out = {}
    for d in sorted(glob.glob(os.path.join(out_root, "*"))):
        cid = os.path.basename(d)
        vals = []
        for f in glob.glob(os.path.join(d, "scores.model_idx_*.npz")):
            z = np.load(f, allow_pickle=True)
            pair = z["per_chain_pair_iptm"]  # (1, n_chain, n_chain)
            vals.append(float(pair[0, protein_chain, ligand_chain]))
        if vals:
            out[cid] = float(np.mean(vals))
    return out


def parse_af3_iptm(out_root: str) -> dict:
    """Map complex id -> interface ipTM from AlphaFold3 outputs.

    AF3 writes ``<id>/<id>_summary_confidences.json`` with a ``chain_pair_iptm`` matrix.
    Assumes chain 0 = protein, chain 1 = ligand. Requires AF3 weights (request-only).
    """
    out = {}
    for f in glob.glob(os.path.join(out_root, "*", "*_summary_confidences.json")):
        cid = os.path.basename(os.path.dirname(f))
        d = json.load(open(f))
        m = d.get("chain_pair_iptm")
        if m and len(m) >= 2:
            out[cid] = float(m[0][1])
    return out


def forge_esmfold2_iptm(fasta_or_inputs, token_env: str = "BIOHUB_TOKEN",
                        model: str = "esmfold2-fast-2026-05"):
    """Score ESMFold2 via the Biohub Forge API. Requires an API token in the environment
    (never pass it inline). Returns id -> interface ipTM.

    This is a thin wrapper; see esm.sdk.esmfold2_client. Provided so ESMFold2 plugs into
    MIRAGE once you supply credentials -- MIRAGE does not handle your token.
    """
    token = os.environ.get(token_env)
    if not token:
        raise RuntimeError(
            f"set {token_env} to your Biohub Forge API token to run ESMFold2 "
            f"(it is a hosted service with no released local inference)."
        )
    raise NotImplementedError(
        "Wire in esm.sdk.esmfold2_client(model=..., token=os.environ['%s']) and extract "
        "the protein-ligand interface ipTM from each MolecularComplexResult." % token_env
    )
