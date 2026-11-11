"""Eligibility-floor card: pipeline only (reference = the pipeline itself, w = 1, no residual clip, no row weights),
per-device levels matched to the best submission's device means, within-device shrinkage spread."""
import subprocess, sys, pandas as pd, numpy as np
U='/home/urad/work/k3/eb/umud'; name=sys.argv[1]; pipe=sys.argv[2]; spr=sys.argv[3:6]
base=pd.read_csv(f'{U}/submissions/d27_S3_mts1014.csv').set_index('image_id')
fe=pd.read_csv(f'{U}/daily/inputs/features_ens.csv').set_index('image_id').loc[base.index]
F=['juniper_right','gray853','h512','telemed_644','left_ticks_800','lumify_png']
fam=fe['family'].astype(str).map(lambda s: next(k for k in F if s.startswith(k)))
fixed=['--w','1','1','1','--pa-offset','1.8374','--alpha','1.0','--no-anchors','--pipeline',pipe,'--ref',pipe,
 '--features','daily/inputs/features_ens.csv','--groups','daily/inputs/test_groups.npy','--sanity','--fail-fill','clip',
 '--aniso','juniper_right','1.12','pa','--aniso','lumify_png','0.87','pa',
 '--spread','pa_deg',spr[0],'--spread','fl_mm',spr[1],'--spread','mt_mm',spr[2]]
lv={f:[0.0,1.0,0.0] for f in F}  # pa add, fl scale, mt add
def run():
    a=list(fixed)
    for f,(pa,fl,mt) in lv.items(): a+=['--fam-pa-final',f,f'{pa:.4f}','--fam-fl-scale',f,f'{fl:.5f}','--fam-mt-offset',f,f'{mt:.4f}']
    r=subprocess.run([sys.executable,'scripts/blend_v2.py',*a,'--out',f'daily/out/{name}.csv'],cwd=U,capture_output=True,text=True)
    if r.returncode: sys.exit(r.stderr)
    open(f'{U}/daily/out/{name}.args','w').write(' '.join(a)); return r.stdout.strip(), pd.read_csv(f'{U}/daily/out/{name}.csv').set_index('image_id').loc[base.index]
for it in range(5):
    msg,o=run()
    m=pd.concat([o.add_suffix('_o'),base.add_suffix('_b')],axis=1).groupby(fam).mean()
    for f in F:
        lv[f][0]+=m.loc[f,'pa_deg_b']-m.loc[f,'pa_deg_o']; lv[f][2]+=m.loc[f,'mt_mm_b']-m.loc[f,'mt_mm_o']; lv[f][1]*=m.loc[f,'fl_mm_b']/m.loc[f,'fl_mm_o']
msg,o=run(); print(msg)
m=pd.concat([o.add_suffix('_o'),base.add_suffix('_b')],axis=1).groupby(fam).mean(); print(m.round(3))
print('nan',o.isna().sum().to_dict())
for t in base.columns: d=o[t]-base[t]; print(t,'mean|d| vs best',round(d.abs().mean(),3))
