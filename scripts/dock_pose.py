"""Dock the pose-benchmark complexes with smina or gnina and score them against the
crystal ligand with symmetry-corrected RMSD (spyrmsd).

The pose arm contrasts two strata of 50 targets each -- singleton families (`novel_bin1`)
and the most redundant families (`seen_bin301`) -- so a docking engine's success rate can
be read as a difficulty floor for the family-support effect seen in the affinity arms.

    python scripts/dock_pose.py smina /path/to/smina \\
        --structures /path/to/PDBbind/P-L --out runs/pose_smina.json

`--structures` is the root of a PDBbind-style tree holding, for each complex,
``<code>/<code>_protein.pdb`` and ``<code>/<code>_ligand.sdf``. The complex directory may
sit directly under the root or one level down (e.g. grouped by redundancy bin); both are
found. Results are written incrementally and already-docked complexes are skipped, so the
run is resumable.
"""
import argparse
import json
import os
import subprocess
import tempfile
from glob import glob
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
POSESET = ROOT / "data" / "pose_targets.json"


def spy(ref_sdf, probe_mols):
    """min symmetry-corrected RMSD between crystal ref and each docked pose"""
    from rdkit import Chem
    from spyrmsd import rmsd as srmsd
    from spyrmsd.molecule import Molecule
    try:
        ref=Molecule.from_rdkit(Chem.SDMolSupplier(ref_sdf, removeHs=True, sanitize=True)[0])
        ref.strip()
        best=None
        for pm in probe_mols:
            if pm is None: continue
            m=Molecule.from_rdkit(pm); m.strip()
            if m.atomicnums.shape[0]!=ref.atomicnums.shape[0]: continue
            r=srmsd.symmrmsd(ref.coordinates, m.coordinates, ref.atomicnums, m.atomicnums,
                             ref.adjacency_matrix, m.adjacency_matrix)
            best=r if best is None else min(best,r)
        return best
    except Exception:
        return None


def complex_dir(base, code):
    """The complex directory, whether it sits at the root or one level down."""
    hits=[d for d in [f"{base}/{code}"]+sorted(glob(f"{base}/*/{code}")) if os.path.isdir(d)]
    return hits[0] if hits else None


def dock(code, base, engine, binary):
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog('rdApp.*')
    d=complex_dir(base, code)
    if d is None: return None
    r=f"{d}/{code}_protein.pdb"; l=f"{d}/{code}_ligand.sdf"
    if not (os.path.exists(r) and os.path.exists(l)): return None
    out=tempfile.mktemp(suffix='.sdf')
    cmd=[binary,'-r',r,'-l',l,'--autobox_ligand',l,'-o',out,'--num_modes','9','--seed','0']
    if engine=='smina': cmd+=['--exhaustiveness','8']
    try:
        subprocess.run(cmd, capture_output=True, timeout=600, env=dict(os.environ))
        if not os.path.exists(out): return None
        poses=[m for m in Chem.SDMolSupplier(out, removeHs=True, sanitize=True)]
        best=spy(l, poses)
        os.remove(out)
        return best
    except Exception:
        return None


def main():
    ap=argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('engine', choices=['smina','gnina'])
    ap.add_argument('binary', help='path to the smina / gnina executable')
    ap.add_argument('--structures', default=os.environ.get('MIRAGE_STRUCTURES'),
                    required='MIRAGE_STRUCTURES' not in os.environ,
                    help='root of the PDBbind-style structure tree (env: MIRAGE_STRUCTURES)')
    ap.add_argument('--poseset', default=str(POSESET),
                    help='JSON with novel_bin1 / seen_bin301 target lists')
    ap.add_argument('--out', help='output JSON (default runs/pose_<engine>.json)')
    a=ap.parse_args()

    out_path=Path(a.out) if a.out else ROOT/"runs"/f"pose_{a.engine}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    res=json.loads(out_path.read_text()) if out_path.exists() else {}

    poseset=json.loads(Path(a.poseset).read_text())
    allc=[(c,'novel') for c in poseset['novel_bin1']]+[(c,'seen') for c in poseset['seen_bin301']]
    todo=[(c,b) for c,b in allc if c not in res]
    print(f"{a.engine}: {len(res)} done, {len(todo)} to dock", flush=True)
    for i,(code,b) in enumerate(todo):
        rm=dock(code, a.structures, a.engine, a.binary)
        if rm is not None: res[code]={'rmsd':rm,'bin':b}
        if (i+1)%5==0:
            out_path.write_text(json.dumps(res)); print(f"  {i+1}/{len(todo)}", flush=True)
    out_path.write_text(json.dumps(res))
    # report per-bin success
    for b in ('novel','seen'):
        v=[x['rmsd'] for x in res.values() if x['bin']==b and x['rmsd'] is not None]
        if v:
            print(f"{a.engine} {b}: n={len(v)} success<2A={np.mean([r<2 for r in v]):.2f} "
                  f"median={np.median(v):.2f}", flush=True)
    print(f"wrote {out_path}", flush=True)


if __name__ == '__main__':
    main()
