"""Regenerate the MIRAGE parquet tables from source data.

This documents exactly how the shipped tables were built; you do not need to run it to
use the benchmark. Sources (download separately, respecting their licenses):

  PDBbind (redundancy set)
    - INDEX_general_PL.<ver>.lst  (affinity labels)
    - per-complex <code>_protein.pdb and <code>_ligand.sdf
  OpenBind A71EV2A (temporal set), CC0
    - github.com/OpenBind-Consortium/A71EV2A-benchmark
    - affinity/reference/fragalysis_compound_reference.csv
    - affinity/predictions/{molecular_weight,clogp,boltz_2}_predictions.csv

Steps:
  1. parse affinity index -> pK (Kd/Ki/IC50, exact '=' only)
  2. extract protein sequence (longest chain CA trace) and ligand SMILES (RDKit)
  3. cluster proteins with MMseqs2 at 30% id / 80% cov -> family_id, family_size
  4. ECFP4 nearest-neighbour Tanimoto -> ligand_nn_tanimoto
  5. balanced 3,360-target 'core' subset (560 per family-size bin)

Requires: rdkit, pandas, pyarrow, and mmseqs2 on PATH. See the code for details.
"""
import argparse
import math
import os
import re
import subprocess
from pathlib import Path

AA3 = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
       "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
       "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
       "TYR": "Y", "VAL": "V", "MSE": "M", "SEC": "C", "PYL": "K"}
SCALE = {"mM": 1e-3, "uM": 1e-6, "nM": 1e-9, "pM": 1e-12, "fM": 1e-15, "M": 1.0}


def parse_index(path):
    rows = {}
    for line in open(path):
        if line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        m = re.search(r"(Kd|Ki|IC50)([=<>~]+)([\d.]+)([munpf]?M)", line)
        if not m or m.group(2) != "=":
            continue
        typ, _, val, unit = m.groups()
        conc = float(val) * SCALE[unit]
        if conc <= 0:
            continue
        rows[parts[0]] = dict(pK=round(-math.log10(conc), 3), atype=typ,
                              year=int(parts[2]) if parts[2].isdigit() else None)
    return rows


def seq_from_pdb(p):
    ch = {}
    for l in open(p, errors="ignore"):
        if l[:4] == "ATOM" and l[12:16].strip() == "CA":
            r = AA3.get(l[17:20].strip())
            if r:
                ch.setdefault(l[21], []).append(r)
    return "".join(max(ch.values(), key=len)) if ch else None


def smiles_from_sdf(p):
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    try:
        m = next(Chem.SDMolSupplier(p, removeHs=True, sanitize=True))
        return (Chem.MolToSmiles(m), m) if m else (None, None)
    except Exception:
        return None, None


def family_bin(s):
    return ("1" if s == 1 else "2-5" if s <= 5 else "6-20" if s <= 20
            else "21-80" if s <= 80 else "81-300" if s <= 300 else "301+")


def build_redundancy(pdbbind_index, complexes_dir, out, workdir):
    import numpy as np
    import pandas as pd
    from rdkit import Chem
    from rdkit.Chem import AllChem

    labels = parse_index(pdbbind_index)
    recs, fps = [], []
    fasta = Path(workdir) / "prot.fasta"
    with open(fasta, "w") as fa:
        for code, lab in labels.items():
            d = Path(complexes_dir) / code
            pp, sp = d / f"{code}_protein.pdb", d / f"{code}_ligand.sdf"
            if not (pp.exists() and sp.exists()):
                continue
            seq = seq_from_pdb(pp)
            smi, mol = smiles_from_sdf(str(sp))
            if not seq or not smi or not (20 <= len(seq) <= 1400):
                continue
            arr = np.zeros(1024, np.uint8)
            Chem.DataStructs.ConvertToNumpyArray(
                AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024), arr)
            fa.write(f">{code}\n{seq}\n")
            recs.append(dict(id=code, sequence=seq, smiles=smi, **lab))
            fps.append(arr)

    # MMseqs2 clustering at 30% id
    cl = Path(workdir) / "clu"
    subprocess.run(["mmseqs", "easy-cluster", str(fasta), str(cl),
                    str(Path(workdir) / "tmp"), "--min-seq-id", "0.3",
                    "-c", "0.8", "--cov-mode", "0"], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    rep = {}
    for line in open(f"{cl}_cluster.tsv"):
        r, m = line.split()
        rep[m] = r
    from collections import Counter
    fam_size = Counter(rep.values())

    # ligand nearest-neighbour Tanimoto
    F = np.array(fps, np.float32)
    card = F.sum(1)
    nn = np.zeros(len(F))
    for i in range(0, len(F), 512):
        b = F[i:i + 512]
        inter = b @ F.T
        tan = inter / np.clip(card[i:i + 512, None] + card[None, :] - inter, 1, None)
        for j in range(b.shape[0]):
            tan[j, i + j] = -1
        nn[i:i + b.shape[0]] = tan.max(1)

    for k, r in enumerate(recs):
        fam = rep.get(r["id"], r["id"])
        r["family_id"] = fam
        r["family_size"] = int(fam_size.get(fam, 1))
        r["redundancy_bin"] = family_bin(r["family_size"])
        r["ligand_nn_tanimoto"] = round(float(nn[k]), 3)
        r["affinity_type"] = r.pop("atype")

    df = pd.DataFrame(recs)
    # balanced core: 560 per bin, spread by ligand novelty, deterministic
    core = set()
    for b in ["1", "2-5", "6-20", "21-80", "81-300", "301+"]:
        sub = df[df.redundancy_bin == b].sort_values(["ligand_nn_tanimoto", "id"])
        step = max(1, len(sub) // 560)
        core |= set(sub.iloc[::step].head(560).id)
    df["in_core"] = df.id.isin(core)
    df.to_parquet(out, index=False)
    print(f"redundancy: {len(df)} rows, {df.family_id.nunique()} families "
          f"-> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdbbind-index", help="INDEX_general_PL.*.lst")
    ap.add_argument("--complexes-dir", help="dir of <code>/<code>_{protein.pdb,ligand.sdf}")
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--workdir", default="/tmp/mirage_build")
    a = ap.parse_args()
    os.makedirs(a.workdir, exist_ok=True)
    os.makedirs(a.out_dir, exist_ok=True)
    if a.pdbbind_index and a.complexes_dir:
        build_redundancy(a.pdbbind_index, a.complexes_dir,
                         os.path.join(a.out_dir, "redundancy.parquet"), a.workdir)
    print("For the temporal set, see the OpenBind A71EV2A repo (CC0); the loader in "
          "scripts already joins reference + baseline CSVs into temporal.parquet.")


if __name__ == "__main__":
    main()
