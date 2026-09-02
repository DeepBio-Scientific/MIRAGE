"""Ligand-only control on the ATOM3D LBA30 benchmark (Sec. LBA30 of the paper).

No published LBA30 result reports a ligand-only / molecular-weight / QSAR baseline.
This script supplies it: train a ligand-only random forest (ECFP4, no protein) on the
LBA30 train split and evaluate Pearson on the test split, alongside molecular-weight,
clogp, and ligand-kNN baselines.

Setup:
    pip install atom3d rdkit scikit-learn scipy
    # download the 30% split (~540 MB) from Zenodo record 4914718:
    #   LBA-split-by-sequence-identity-30.tar.gz  -> untar into $LBA_DIR
    python scripts/lba30_ligand_only.py --lba-dir $LBA_DIR/split-by-sequence-identity-30

The official split is 3507 train / 490 test. Not every LMDB entry carries a usable
``smiles`` field, so a naive ligand-only control silently evaluates on a subset (453 of
490), which is not the set the published deep-net numbers are computed on. This script
therefore falls back to the ligand atom block: molecular weight is summed directly from
the elements (so it covers all 490 entries and is comparable to the literature numbers),
and a connectivity-perceived molecule supplies ECFP4/clogp where SMILES is absent. The
coverage of every reported baseline is printed, and any residual shortfall is named.

Reference values on the recovered set: ligand-only RF 0.457, molecular weight 0.438,
ligand-kNN 0.429, clogp 0.212 -- vs published IPBind 0.732, EHIGN 0.612, 3DCNN 0.550,
DeepDTA 0.472, ENN 0.389 (all on n=490).
"""
import argparse
import numpy as np
from scipy import stats


def load_split(lba_dir, split):
    """Return (id, smiles_or_None, atoms_ligand_or_None, pK) for every labelled entry."""
    from atom3d.datasets import LMDBDataset
    ds = LMDBDataset(f"{lba_dir}/data/{split}")
    rows = []
    for it in ds:
        y = it.get("scores", {}).get("neglog_aff")
        if y is None:
            continue
        rows.append((it["id"], it.get("smiles"), it.get("atoms_ligand"), float(y)))
    return rows


def mol_from_atoms(atoms):
    """Perceive a molecule from the ligand atom block when SMILES is missing."""
    from rdkit import Chem
    from rdkit.Chem import rdDetermineBonds
    if atoms is None or len(atoms) == 0:
        return None
    el = [str(e).capitalize() for e in atoms["element"]]
    xyz = [f"{len(el)}\n\n"] + [
        f"{e} {x:.4f} {y:.4f} {z:.4f}\n"
        for e, x, y, z in zip(el, atoms["x"], atoms["y"], atoms["z"])]
    try:
        m = Chem.MolFromXYZBlock("".join(xyz))
        if m is None:
            return None
        rdDetermineBonds.DetermineConnectivity(m)
        Chem.SanitizeMol(m)
        return m
    except Exception:
        return None


def weight_from_atoms(atoms):
    """Molecular weight straight from the element list -- needs no bond perception,
    so it is defined for every entry in the split."""
    from rdkit.Chem import GetPeriodicTable
    if atoms is None or len(atoms) == 0:
        return float("nan")
    pt = GetPeriodicTable()
    try:
        return float(sum(pt.GetAtomicWeight(str(e).capitalize()) for e in atoms["element"]))
    except Exception:
        return float("nan")


def featurize(rows, n_bits=2048):
    """Featurize every labelled entry. Rows without a usable molecule keep a NaN
    fingerprint/clogp but still carry a molecular weight from the atom block."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem, Descriptors, Crippen, DataStructs
    RDLogger.DisableLog("rdApp.*")
    X, y, mw, cl, from_smiles = [], [], [], [], 0
    for _, smi, atoms, yy in rows:
        m = Chem.MolFromSmiles(smi) if smi else None
        from_smiles += m is not None
        if m is None:
            m = mol_from_atoms(atoms)
        if m is None:
            a = np.full(n_bits, np.nan)
            cl.append(float("nan"))
        else:
            a = np.zeros(n_bits, np.uint8)
            DataStructs.ConvertToNumpyArray(
                AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=n_bits), a)
            cl.append(Crippen.MolLogP(m))
        w = Descriptors.MolWt(m) if m is not None else weight_from_atoms(atoms)
        X.append(a); y.append(yy); mw.append(w)
    return (np.array(X, float), np.array(y), np.array(mw), np.array(cl), from_smiles)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lba-dir", required=True,
                    help="path to .../split-by-sequence-identity-30")
    a = ap.parse_args()
    from sklearn.ensemble import RandomForestRegressor

    tr = load_split(a.lba_dir, "train")
    te = load_split(a.lba_dir, "test")
    Xtr, ytr, _, _, tr_smi = featurize(tr)
    Xte, yte, mwte, clte, te_smi = featurize(te)
    tr_ok = np.isfinite(Xtr).all(1)
    print(f"labelled entries: train {len(ytr)} (official 3507), test {len(yte)} (official 490)")
    print(f"  with a SMILES field       : train {tr_smi}, test {te_smi}")
    print(f"  molecule recovered in all : train {int(tr_ok.sum())}, "
          f"test {int(np.isfinite(Xte).all(1).sum())}")

    rf = RandomForestRegressor(n_estimators=500, min_samples_leaf=2, n_jobs=-1, random_state=0)
    rf.fit(Xtr[tr_ok], ytr[tr_ok])

    fp_ok = np.isfinite(Xte).all(1)
    p = np.full(len(yte), np.nan)
    p[fp_ok] = rf.predict(Xte[fp_ok])
    Xtr_ok = Xtr[tr_ok]; ytr_ok = ytr[tr_ok]
    card = Xtr_ok.sum(1); knn = np.full(len(yte), np.nan)
    for i in np.flatnonzero(fp_ok):
        inter = Xte[i] @ Xtr_ok.T
        tan = inter / np.clip(Xte[i].sum() + card - inter, 1, None)
        nn = np.argsort(-tan)[:5]; w = np.clip(tan[nn], 1e-6, None)
        knn[i] = (w * ytr_ok[nn]).sum() / w.sum()

    def report(label, v):
        ok = np.isfinite(v) & np.isfinite(yte)
        r = stats.pearsonr(v[ok], yte[ok])[0]
        flag = "" if ok.sum() == len(yte) else f"  <- {len(yte)-int(ok.sum())} entries not covered"
        print(f"  {label:22s}: Pearson {r:+.3f}  (n={int(ok.sum())}){flag}")

    print("\nLBA30 ligand-only controls:")
    report("ligand-only RF (ECFP4)", p)
    report("molecular weight", mwte)
    report("ligand-kNN", knn)
    report("clogp", clte)
    print("  reference (n=490): IPBind 0.732 | EHIGN 0.612 | 3DCNN 0.550 | "
          "DeepDTA 0.472 | ENN 0.389")
    print("\nReport the n for each control next to the literature n; a control evaluated "
          "on fewer complexes is not a like-for-like comparison.")


if __name__ == "__main__":
    main()
