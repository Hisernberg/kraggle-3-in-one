import numpy as np, pandas as pd
W='/home/user/work'
d=pd.read_csv('ext_flv3_merged.csv')
x=pd.read_excel(f'{W}/osf/benchmark_arch/benchmark_dataset_architecture_v0.1.0/Results_benchmark_architecture_v0.1.0.xlsx',sheet_name='Manual_architecture').rename(columns={'ImageID':'image_id'})
d=d.merge(x[['image_id','DLTrack_FL']],on='image_id',how='left')
d['cwm_fb']=d.fl_cwmed.fillna(d.fl_med)
d['cvis3_fb']=d.fl_cvis3.fillna(d.fl_med)
d['new_avg']=d[['cwm_fb','v2_fl_chord_mid']].mean(axis=1)
d['new_avg3']=d[['cwm_fb','fl_med','v2_fl_chord_mid']].mean(axis=1)
d['vis5_avg']=d[['fl_vis5','v2_fl_chord_mid']].mean(axis=1)
rng=np.random.default_rng(0)
def paired(m,a,b,deb=True):
    ok=m[a].notna()&m[b].notna()&m.fl_mm.notna(); m=m[ok]
    ea=m[a]-m.fl_mm; eb=m[b]-m.fl_mm
    if deb: ea=ea-ea.mean(); eb=eb-eb.mean()
    diff=(eb.abs()-ea.abs()).values
    bs=[rng.choice(diff,len(diff)).mean() for _ in range(4000)]
    return len(m), ea.abs().mean(), eb.abs().mean(), np.percentile(bs,[2.5,97.5]), np.corrcoef(ea,eb)[0,1]
for s in ['osf','neuage','gm']:
    m=d[d.set==s]
    for b in ['cwm_fb','new_avg','new_avg3','vis5_avg','cvis3_fb']:
        n,ma,mb,ci,r=paired(m,'cur',b)
        print(f'{s:6s} cur vs {b:9s} n={n:3d} debiased {ma:6.2f} -> {mb:6.2f}  diff CI [{ci[0]:+.2f},{ci[1]:+.2f}]  err corr {r:.2f}')
# OSF blended sim with DLTrack (debiased sources), FL weight 0.31
m=d[d.set=='osf'].copy()
for col in ['cur','cwm_fb','new_avg','new_avg3','vis5_avg']:
    P=m[col]-(m[col]-m.fl_mm).mean(); D=m.DLTrack_FL-(m.DLTrack_FL-m.fl_mm).mean()
    res={w: (D+w*(P-D)-m.fl_mm).abs().mean() for w in [0.2,0.31,0.45]}
    print('OSF blend sim',col,{k:round(v,2) for k,v in res.items()}, 'corr(P err, D err) %.2f'%np.corrcoef(P-m.fl_mm,D-m.fl_mm)[0,1])
