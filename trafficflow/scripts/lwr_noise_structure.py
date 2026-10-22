import glob, numpy as np, pandas as pd, sys
R='/home/user/data/kaggle_public/corridors/'
P=sys.argv[1] if len(sys.argv)>1 else 'D12_I5_N'
fs=sorted(glob.glob(f'{R}{P}/train/mainline_states/*/*.parquet'))[:40]
d=pd.concat([pd.read_parquet(f,columns=['timestamp','link_id','speed_kmh','flow_vph','occupancy','pct_observed','is_score_eligible']) for f in fs])
d=d[d.is_score_eligible.astype(bool)].groupby(['timestamp','link_id'],as_index=False).mean(numeric_only=True)
V=d.pivot(index='timestamp',columns='link_id',values='speed_kmh').sort_index()
Q=d.pivot(index='timestamp',columns='link_id',values='flow_vph').reindex(V.index)
O=d.pivot(index='timestamp',columns='link_id',values='occupancy').reindex(V.index)
v,q,o=V.values,Q.values,O.values
k=q/np.maximum(v,1)
def res(x):  # residual from neighbours mean in time
    return x[1:-1]-(x[:-2]+x[2:])/2
free=(v[1:-1]>90)&(v[:-2]>90)&(v[2:]>90)
for name,x in [('logv',np.log(v)),('logq',np.log(q)),('logk',np.log(k)),('logocc',np.log(np.maximum(o,1e-3)))]:
    r=res(x); print(name,'res sd all %.4f free %.4f cong %.4f'%(np.nanstd(r),np.nanstd(r[free]),np.nanstd(r[~free])))
rv,rq,ro=res(np.log(v)),res(np.log(q)),res(np.log(np.maximum(o,1e-3)))
m=np.isfinite(rv)&np.isfinite(rq)&np.isfinite(ro)
print('corr res logv-logq', np.corrcoef(rv[m],rq[m])[0,1], 'free',np.corrcoef(rv[m&free],rq[m&free])[0,1])
rk=res(np.log(k)); mk=m&np.isfinite(rk)
print('corr res logk-logocc', np.corrcoef(rk[mk],ro[mk])[0,1], 'logq-logocc',np.corrcoef(rq[m],ro[m])[0,1])
# lag-1 autocorr of first differences in k
dk=np.diff(k,axis=0); a,b=dk[:-1].ravel(),dk[1:].ravel(); mm=np.isfinite(a)&np.isfinite(b)
print('lag1 corr dk',np.corrcoef(a[mm],b[mm])[0,1])
print('mean |dk|/k',np.nanmean(np.abs(dk))/np.nanmean(k))
# relation k vs occ
mm=np.isfinite(k)&np.isfinite(o)&(o>0)
ln=np.log(k[mm]/o[mm]); print('k/occ ratio sd(log)',np.std(ln), 'median',np.exp(np.median(ln)))
# spatial correlation of time-residuals at same t (links sorted by milepost)
mp=d.groupby('link_id').size()  # placeholder
f0=pd.read_parquet(fs[0],columns=['link_id','milepost']).groupby('link_id').milepost.mean()
order=f0.reindex(V.columns).sort_values().index
idx=[list(V.columns).index(c) for c in order]
for nm,x in [('logq',np.log(q)),('logv',np.log(v)),('logk',np.log(k))]:
    r=res(x)[:,idx]
    for lag in (1,2,5):
        a,b=r[:,:-lag].ravel(),r[:,lag:].ravel(); mm=np.isfinite(a)&np.isfinite(b)
        print(nm,'spatial lag',lag,'corr %.3f'%np.corrcoef(a[mm],b[mm])[0,1])
# time autocorr of deviation from 13-step rolling median
for nm,x in [('logq',np.log(q)),('logk',np.log(k))]:
    s=pd.DataFrame(x).rolling(13,center=True,min_periods=5).median().values; e=x-s
    for lag in (1,2,3,6):
        a,b=e[:-lag].ravel(),e[lag:].ravel(); mm=np.isfinite(a)&np.isfinite(b)
        print(nm,'time lag',lag,'acf %.3f'%np.corrcoef(a[mm],b[mm])[0,1])
