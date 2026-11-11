"""Build a card with run.py, then rescale per-family FL so every device's mean final FL equals the base card's."""
import subprocess, sys, json, pandas as pd, numpy as np
U='/home/urad/work/k3/eb/umud'; name=sys.argv[1]; extra=sys.argv[2:]
base=pd.read_csv(f'{U}/submissions/d29_S1_jlpa_vdljl0.csv').set_index('image_id')
fe=pd.read_csv(f'{U}/daily/inputs/features_ens.csv').set_index('image_id').loc[base.index]
fam=fe['family'].astype(str).map(lambda s: next(k for k in ['juniper_right','gray853','h512','telemed_644','left_ticks_800','lumify_png'] if s.startswith(k)))
st=json.load(open(f'{U}/daily/state.json'))['best_args']
sc={st[i+1]:float(st[i+2]) for i,t in enumerate(st) if t=='--fam-pa-final'}
def run(sc):
    ex=list(extra)
    for k,v in sc.items(): ex+=['--fam-pa-final',k,f'{v:.4f}']
    r=subprocess.run([sys.executable,'daily/run.py','build',name,*ex],cwd=U,capture_output=True,text=True); 
    return pd.read_csv(f'{U}/daily/out/{name}.csv').set_index('image_id').loc[base.index]
for it in range(4):
    o=run(sc)
    m=pd.DataFrame({'f':fam,'o':o.pa_deg,'b':base.pa_deg}).groupby('f')[['o','b']].mean()
    print(it,(m.o-m.b).round(4).to_dict())
    for k in sc: sc[k]+=m.loc[k,'b']-m.loc[k,'o']
o=run(sc)
for t in ['pa_deg','pa_deg','mt_mm']:
    d=o[t]-base[t]; print(t,'rows moved',int((d.abs()>1e-6).sum()),'sd',round(d.std(),3),'mean',round(d.mean(),4))
print(open(f'{U}/daily/out/{name}.args').read())
