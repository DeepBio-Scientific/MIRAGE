"""Reproduce the RF-QSAR and ligand-kNN baselines from the paper.

Both are evaluated under 5-fold FAMILY-DISJOINT cross-validation (GroupKFold on
family_id), so every prediction is on a held-out protein family -- the honest
counterpart to co-folders whose test families are present in their PDB training data.

Features:
  ligand  = ECFP4, 1024 bits (RDKit Morgan radius 2)
  protein = amino-acid composition (20) + log10 length

Outputs id,prediction CSVs compatible with `mirage score redundancy ...`.
Requires: rdkit, scikit-learn, pandas, numpy. Run after loading the redundancy set.
"""
import argparse

import numpy as np
import pandas as pd


def ecfp(smiles, n_bits=1024):
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem
    RDLogger.DisableLog("rdApp.*")
    m = Chem.MolFromSmiles(smiles)
    arr = np.zeros(n_bits, np.uint8)
    if m is not None:
        Chem.DataStructs.ConvertToNumpyArray(
            AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=n_bits), arr)
    return arr


AAS = "ACDEFGHIKLMNPQRSTVWY"


def aacomp(seq):
    n = len(seq)
    c = np.array([seq.count(a) for a in AAS], float)
    return np.append(c / max(n, 1), np.log10(n + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="dir with redundancy.parquet")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--folds", type=int, default=5)
    a = ap.parse_args()

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import GroupKFold

    import mirage
    df = mirage.load("redundancy", path=a.data).reset_index(drop=True)
    print(f"[train] featurising {len(df)} complexes...")
    L = np.vstack([ecfp(s) for s in df.smiles]).astype(np.float32)
    P = np.vstack([aacomp(s) for s in df.sequence])
    X = np.hstack([L, P])
    y = df.pK.values
    g = df.family_id.values
    gkf = GroupKFold(a.folds)

    # RF-QSAR
    rf = np.full(len(y), np.nan)
    for k, (tr, te) in enumerate(gkf.split(X, y, g)):
        m = RandomForestRegressor(n_estimators=300, min_samples_leaf=3,
                                  n_jobs=-1, random_state=0)
        m.fit(X[tr], y[tr])
        rf[te] = m.predict(X[te])
        print(f"[train] RF fold {k + 1}/{a.folds}")
    pd.DataFrame({"id": df.id, "prediction": rf.round(4)}).to_csv(
        f"{a.out_dir}/rf_qsar_redundancy.csv", index=False)

    # ligand-kNN (family-disjoint, k=5, ECFP Tanimoto)
    card = L.sum(1)
    knn = np.full(len(y), np.nan)
    K = 5
    for tr, te in gkf.split(X, y, g):
        Ltr, ctr, ytr = L[tr], card[tr], y[tr]
        for c0 in range(0, len(te), 256):
            idx = te[c0:c0 + 256]
            b = L[idx]
            inter = b @ Ltr.T
            tan = inter / np.clip(card[idx, None] + ctr[None, :] - inter, 1, None)
            nn = np.argsort(-tan, axis=1)[:, :K]
            w = np.clip(np.take_along_axis(tan, nn, axis=1), 1e-6, None)
            knn[idx] = (w * ytr[nn]).sum(1) / w.sum(1)
    pd.DataFrame({"id": df.id, "prediction": knn.round(4)}).to_csv(
        f"{a.out_dir}/ligand_knn_redundancy.csv", index=False)
    print(f"[train] wrote rf_qsar_redundancy.csv and ligand_knn_redundancy.csv "
          f"to {a.out_dir}")


if __name__ == "__main__":
    main()
