"""Build a card with run.py and level-match per-device means of one target to a base submission.
usage: levelmatch_any.py BASE_CSV TARGET(pa|fl) NAME [extra blend args]
PA: additive --fam-pa-final per device; FL: multiplicative --fam-fl-scale per device (start values from state best_args)."""
import subprocess, sys, json, pandas as pd
U='/home/urad/work/k3/eb/umud'; basep, tgt, name, extra = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
base=pd.read_csv(basep).set_index('image_id')
fe=pd.read_csv(f'{U}/daily/inputs/features_ens.csv').set_index('image_id').loc[base.index]
FAMS=['juniper_right','gray853','h512','telemed_644','left_ticks_800','lumify_png']
fam=fe['family'].astype(str).map(lambda s: next(k for k in FAMS if s.startswith(k)))
st=json.load(open(f'{U}/daily/state.json'))['best_args']
flag, col = ('--fam-pa-final','pa_deg') if tgt=='pa' else ('--fam-fl-scale','fl_mm')
sc={st[i+1]:float(st[i+2]) for i,t in enumerate(st) if t==flag}
def run(sc):
    ex=list(extra)
    for k,v in sc.items(): ex+=[flag,k,(f'{v:.4f}' if tgt=='pa' else f'{v:.5f}')]
    r=subprocess.run([sys.executable,'daily/run.py','build',name,*ex],cwd=U,capture_output=True,text=True)
    if r.returncode: print(r.stderr); sys.exit(1)
    return pd.read_csv(f'{U}/daily/out/{name}.csv').set_index('image_id').loc[base.index]
for it in range(5):
    o=run(sc)
    m=pd.DataFrame({'f':fam,'o':o[col],'b':base[col]}).groupby('f')[['o','b']].mean()
    print(it,(m.o-m.b).round(4).to_dict())
    for k in sc:
        if tgt=='pa': sc[k]+=m.loc[k,'b']-m.loc[k,'o']
        else: sc[k]*=m.loc[k,'b']/m.loc[k,'o']
o=run(sc)
for t in ['pa_deg','fl_mm','mt_mm']:
    d=o[t]-base[t]; print(t,'rows moved',int((d.abs()>1e-6).sum()),'mean|d|',round(d.abs().mean(),4),'mean',round(d.mean(),4))
print(open(f'{U}/daily/out/{name}.args').read())
