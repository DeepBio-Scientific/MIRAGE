import json, numpy as np, pandas as pd
from scipy import stats
from sklearn.model_selection import KFold
red=pd.read_parquet('/home/mehdi/dev/fambench/data/redundancy.parquet').reset_index(drop=True).rename(columns={'id':'code'})
red['loglen']=np.log10(red.sequence.str.len())
M='/home/mehdi/dev/mrnafold/data/interim/'
preds={
 'Nesso-1':{r['code']:r['paff'] for r in json.load(open(M+'nesso_leakage_join.json'))},
 'Boltz-2':{r['code']:r['paff'] for r in json.load(open(M+'boltz_leakage_join.json'))},
 'RF-QSAR':json.load(open(M+'pred_rf_qsar.json')),
 'ligand-kNN':json.load(open(M+'pred_ligknn.json')),
 'mol.weight':json.load(open(M+'pred_mw.json')),
 'clogp':json.load(open(M+'pred_clogp.json')),
}
def fbin(s): return '1' if s==1 else '2-5' if s<=5 else '6-20' if s<=20 else '21-80' if s<=80 else '81-300' if s<=300 else '301+'
red['fbin']=red.family_size.map(fbin)

def prep(df, pred, b):
    d=df[(df.fbin==b)&(df.pK>=4.5)&(df.pK<=8.0)&(df.code.isin(pred))]
    y=d.pK.values; p=np.array([pred[c] for c in d.code]); fam=d.family_id.values
    # group indices by family
    fam2idx={}
    for i,f in enumerate(fam): fam2idx.setdefault(f,[]).append(i)
    fams=list(fam2idx); groups=[np.array(fam2idx[f]) for f in fams]
    return y,p,groups

def r_of(y,p,idx):
    yy,pp=y[idx],p[idx]
    return stats.pearsonr(pp,yy)[0] if yy.std()>0 and pp.std()>0 else np.nan

def boot_gap(df,pred,n=500,seed=0):
    rng=np.random.default_rng(seed)
    yh,ph,gh=prep(df,pred,'301+'); yl,pl,gl=prep(df,pred,'1')
    def one(y,p,groups):
        pick=rng.integers(0,len(groups),len(groups))
        idx=np.concatenate([groups[k] for k in pick])
        # resample ligands within (already concatenated families) — second level
        idx=idx[rng.integers(0,len(idx),len(idx))]
        return r_of(y,p,idx)
    g=[]
    for _ in range(n):
        a=one(yh,ph,gh); b=one(yl,pl,gl)
        if np.isfinite(a) and np.isfinite(b): g.append(a-b)
    return np.percentile(g,[2.5,97.5])

def matched_r(df,pred,b):
    y,p,_=prep(df,pred,b); return r_of(y,p,np.arange(len(y)))

G={}
for m,pred in preds.items():
    rh=matched_r(red,pred,'301+'); rl=matched_r(red,pred,'1'); ci=boot_gap(red,pred)
    G[m]=dict(G=float(rh-rl),r_high=float(rh),r_low=float(rl),ci=[float(ci[0]),float(ci[1])])
    print(f"{m:12s} G={rh-rl:+.3f} [r(301+)={rh:+.3f} r(1)={rl:+.3f}] 95%CI[{ci[0]:+.2f},{ci[1]:+.2f}]",flush=True)

# family-mean baseline
y=red.pK.values; fid=red.family_id.values; oof=np.full(len(y),np.nan)
for tr,te in KFold(5,shuffle=True,random_state=0).split(y):
    fm=pd.Series(y[tr]).groupby(fid[tr]).mean(); gm=y[tr].mean()
    oof[te]=[fm.get(fid[i],gm) for i in te]
rfm=float(stats.pearsonr(oof,y)[0])
print(f"family-mean random: {rfm:+.3f}",flush=True)
json.dump({'G':G,'family_mean_random':rfm}, open(M+'paper_gap_analysis.json','w'), indent=1)
print("GAP_DONE",flush=True)
