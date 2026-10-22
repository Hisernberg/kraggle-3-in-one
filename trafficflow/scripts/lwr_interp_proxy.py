import glob, numpy as np, pandas as pd, sys
R='/home/user/data/kaggle_public/corridors/'
P=sys.argv[1] if len(sys.argv)>1 else 'D12_I5_N'
fs=sorted(glob.glob(f'{R}{P}/train/mainline_states/*/*.parquet'))[100:130]
out=[]
rng=np.random.default_rng(0)
def interp_nan(x):  # time-linear interpolation per column, edges ffill/bfill
    x=x.copy(); T=x.shape[0]; t=np.arange(T)
    for j in range(x.shape[1]):
        c=x[:,j]; m=np.isfinite(c)
        if m.sum()>=2: x[:,j]=np.interp(t,t[m],c[m])
    return x
for f in fs:
    d=pd.read_parquet(f,columns=['timestamp','link_id','speed_kmh','flow_vph','is_score_eligible'])
    d=d[d.is_score_eligible.astype(bool)].groupby(['timestamp','link_id'],as_index=False).mean(numeric_only=True)
    V=d.pivot(index='timestamp',columns='link_id',values='speed_kmh').sort_index().values
    Q=d.pivot(index='timestamp',columns='link_id',values='flow_vph').sort_index().values
    for p in (0.2,0.3,0.5):
        tg=(rng.random(V.shape)<p)&np.isfinite(V)
        k=Q/np.maximum(V,1)
        Vm,Qm,km=V.copy(),Q.copy(),k.copy(); Vm[tg]=np.nan;Qm[tg]=np.nan;km[tg]=np.nan
        Vi,Qi,ki=interp_nan(Vm),interp_nan(Qm),interp_nan(km)
        dN=np.diff(k,axis=0); ok=np.isfinite(dN)
        tt=(tg[1:]|tg[:-1])&ok
        den=np.abs(dN[ok]).sum()
        res={}
        for nm,kk in [('qv_interp',Qi/np.maximum(Vi,1)),('k_interp',ki),('const_prev',None)]:
            if kk is None: continue
            ks=np.where(tg,kk,k); r=np.abs(np.diff(ks,axis=0)-dN)[ok].sum()
            res[nm]=1-r/den
        res['frac_target_trans']=np.abs(dN[tt]).sum()/den
        out.append(dict(p=p,**res))
o=pd.DataFrame(out); print(P); print(o.groupby('p').mean()); print(o.drop(columns='p').mean())
