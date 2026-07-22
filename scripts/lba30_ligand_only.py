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

Result (n_test=453): ligand-only RF 0.457, molecular weight 0.438, ligand-kNN 0.429,
clogp 0.212 -- vs published IPBind 0.732, EHIGN 0.612, 3DCNN 0.550, DeepDTA 0.472, ENN 0.389.
"""
import argparse
import numpy as np
from scipy import stats


def load_split(lba_dir, split):
    from atom3d.datasets import LMDBDataset
    ds = LMDBDataset(f"{lba_dir}/data/{split}")
    rows = []
    for it in ds:
        smi = it.get("smiles")
        y = it["scores"]["neglog_aff"]
        if smi and y is not None:
            rows.append((it["id"], smi, float(y)))
    return rows


def featurize(rows, n_bits=2048):
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem, Descriptors, Crippen, DataStructs
    RDLogger.DisableLog("rdApp.*")
    X, y, mw, cl = [], [], [], []
    for _, smi, yy in rows:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            continue
        a = np.zeros(n_bits, np.uint8)
        DataStructs.ConvertToNumpyArray(
            AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=n_bits), a)
        X.append(a); y.append(yy); mw.append(Descriptors.MolWt(m)); cl.append(Crippen.MolLogP(m))
    return np.array(X, float), np.array(y), np.array(mw), np.array(cl)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lba-dir", required=True,
                    help="path to .../split-by-sequence-identity-30")
    a = ap.parse_args()
    from sklearn.ensemble import RandomForestRegressor

    tr = load_split(a.lba_dir, "train")
    te = load_split(a.lba_dir, "test")
    Xtr, ytr, _, _ = featurize(tr)
    Xte, yte, mwte, clte = featurize(te)
    print(f"train {len(ytr)}, test {len(yte)}")

    rf = RandomForestRegressor(n_estimators=500, min_samples_leaf=2, n_jobs=-1, random_state=0)
    rf.fit(Xtr, ytr)
    p = rf.predict(Xte)
    card = Xtr.sum(1); ct = Xte.sum(1); knn = np.zeros(len(yte))
    for i in range(len(yte)):
        inter = Xte[i] @ Xtr.T
        tan = inter / np.clip(ct[i] + card - inter, 1, None)
        nn = np.argsort(-tan)[:5]; w = np.clip(tan[nn], 1e-6, None)
        knn[i] = (w * ytr[nn]).sum() / w.sum()

    print(f"\nLBA30 ligand-only controls (test n={len(yte)}):")
    print(f"  ligand-only RF (ECFP4) : Pearson {stats.pearsonr(p, yte)[0]:+.3f}  "
          f"RMSE {np.sqrt(((p-yte)**2).mean()):.3f}")
    print(f"  molecular weight       : Pearson {stats.pearsonr(mwte, yte)[0]:+.3f}")
    print(f"  ligand-kNN             : Pearson {stats.pearsonr(knn, yte)[0]:+.3f}")
    print(f"  clogp                  : Pearson {stats.pearsonr(clte, yte)[0]:+.3f}")
    print("  reference: IPBind 0.732 | EHIGN 0.612 | 3DCNN 0.550 | DeepDTA 0.472 | ENN 0.389")


if __name__ == "__main__":
    main()
