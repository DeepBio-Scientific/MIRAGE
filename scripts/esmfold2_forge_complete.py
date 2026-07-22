"""Resumable ESMFold2 (Forge) completion pass for the FamBench confidence-proxy arm.

Forge caps at ~100 folds/day, so completing the 600-target balanced set takes several
daily runs. This script SKIPS already-folded complexes and stops gracefully at the daily
credit cap, so run it once per day (e.g. via crontab) until it reports 600/600:

    conda activate esm312 && python scripts/esmfold2_forge_complete.py

Reads BIOHUB_API_KEY from ../mrnafold/.env (never printed). Writes incremental results to
data/interim/pred_esmfold2_iptm.json. Note: at n=281 the ESMFold2 ipTM curve is already
flat (no family-support dependence, t=-0.7); completing to 600 is for thoroughness and is
not expected to change that conclusion.
"""
import os, json, time
import pandas as pd
for line in open('/home/mehdi/dev/mrnafold/.env'):
    if line.strip().startswith('BIOHUB_API_KEY='):
        tok=line.strip().split('=',1)[1].strip().strip('"').strip("'")
from esm.sdk import esmfold2_client
from esm.sdk.api import FoldingConfig
from esm.utils.structure.input_builder import ProteinInput, LigandInput, StructurePredictionInput
client=esmfold2_client(token=tok)
cfg=FoldingConfig(num_loops=3, num_sampling_steps=30, include_pae=True)

seqs={}; c=None
for l in open('/home/mehdi/dev/mrnafold/data/interim/pdbbind_prot.fasta'):
    if l.startswith('>'): c=l[1:].strip()
    else: seqs[c]=l.strip()
red=pd.read_parquet('/home/mehdi/dev/fambench/data/redundancy.parquet')
smiles=dict(zip(red.id,red.smiles))
codes=json.load(open('/tmp/fambench_scratch/chai_codes.json'))  # the full 600 balanced set

OUT='/home/mehdi/dev/mrnafold/data/interim/pred_esmfold2_iptm.json'
out=json.load(open(OUT)) if os.path.exists(OUT) else {}
todo=[c for c in codes if c not in out and c in seqs and c in smiles]
print(f"have {len(out)}, remaining to complete 600: {len(todo)}", flush=True)

done=0; capped=False; consec=0; t0=time.time()
for i,code in enumerate(todo):
    try:
        inp=StructurePredictionInput(sequences=[
            ProteinInput(id="A", sequence=seqs[code][:1000]),
            LigandInput(id="B", ccd=None, smiles=smiles[code])])
        r=client.fold_all_atom(inp, config=cfg)
        pci=getattr(r,'pair_chains_iptm',None); val=None
        if isinstance(pci,dict):
            try: val=float(pci['A']['B'])
            except Exception: val=None
        if val is None: val=float(r.iptm)
        out[code]=val; done+=1; consec=0
        json.dump(out, open(OUT,'w'))
    except Exception as e:
        consec+=1
        if consec>=5:
            print(f"CAP/ERROR: {consec} consecutive failures after +{done} (total {len(out)}/600); stopping", flush=True); capped=True; break
    time.sleep(3.2)
    if (i+1)%10==0: print(f"{i+1}/{len(todo)} +{done} {time.time()-t0:.0f}s", flush=True)
json.dump(out, open(OUT,'w'))
print(f"PASS DONE: +{done} new, total {len(out)}/600, capped={capped}", flush=True)
print("COMPLETE_DONE" if len(out)>=595 else ("CAPPED" if capped else "PASS_END"), flush=True)
