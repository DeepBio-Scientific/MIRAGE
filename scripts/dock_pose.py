"""Dock pose-benchmark complexes with smina or gnina, compute spyrmsd to crystal.
Usage: python dock_pose.py <engine: smina|gnina> <gnina_binary_or_smina>"""
import json, os, sys, subprocess, tempfile
import numpy as np
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')
from spyrmsd import rmsd as srmsd
from spyrmsd.molecule import Molecule

ENGINE=sys.argv[1]; BIN=sys.argv[2]
LDP=os.environ.get('LD_LIBRARY_PATH','')
BASE='/tmp/fambench_scratch/plall/P-L'
aff=json.load(open('/home/mehdi/dev/mrnafold/data/interim/pdbbind_affinity.json'))
binof={a['code']:a['bin'] for a in aff}
poseset=json.load(open('/home/mehdi/dev/mrnafold/data/interim/poseset.json'))
OUT=f'/home/mehdi/dev/mrnafold/data/interim/pose_{ENGINE}.json'
res=json.load(open(OUT)) if os.path.exists(OUT) else {}

def spy(ref_sdf, probe_mols):
    """min symmetry-corrected RMSD between crystal ref and each docked pose"""
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

def dock(code):
    d=f"{BASE}/{binof[code]}/{code}"; r=f"{d}/{code}_protein.pdb"; l=f"{d}/{code}_ligand.sdf"
    if not (os.path.exists(r) and os.path.exists(l)): return None
    out=tempfile.mktemp(suffix='.sdf')
    cmd=[BIN,'-r',r,'-l',l,'--autobox_ligand',l,'-o',out,'--num_modes','9','--seed','0']
    if ENGINE=='smina': cmd+=['--exhaustiveness','8']
    env=dict(os.environ); env['LD_LIBRARY_PATH']=LDP
    try:
        subprocess.run(cmd, capture_output=True, timeout=600, env=env)
        if not os.path.exists(out): return None
        poses=[m for m in Chem.SDMolSupplier(out, removeHs=True, sanitize=True)]
        best=spy(l, poses)
        os.remove(out)
        return best
    except Exception:
        return None

allc=[(c,'novel') for c in poseset['novel_bin1']]+[(c,'seen') for c in poseset['seen_bin301']]
todo=[(c,b) for c,b in allc if c not in res]
print(f"{ENGINE}: {len(res)} done, {len(todo)} to dock", flush=True)
for i,(code,b) in enumerate(todo):
    rm=dock(code)
    if rm is not None: res[code]={'rmsd':rm,'bin':b}
    if (i+1)%5==0:
        json.dump(res, open(OUT,'w')); print(f"  {i+1}/{len(todo)}", flush=True)
json.dump(res, open(OUT,'w'))
# report per-bin success
for b in ('novel','seen'):
    v=[x['rmsd'] for x in res.values() if x['bin']==b and x['rmsd'] is not None]
    if v: print(f"{ENGINE} {b}: n={len(v)} success<2A={np.mean([r<2 for r in v]):.2f} median={np.median(v):.2f}", flush=True)
print(f"{ENGINE.upper()}_POSE_DONE", flush=True)
