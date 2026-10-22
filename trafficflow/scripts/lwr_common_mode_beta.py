# Does J2 already use the same-time spatial common mode? compare with the train-optimal beta.
import glob, numpy as np, pandas as pd, sys
R='/home/user/data/kaggle_public/corridors/'
P=sys.argv[1]
def arrs(df, links):
    g=df.groupby(['timestamp','link_id'],as_index=False)[['speed_kmh','flow_vph']].mean()
    ts=pd.date_range(df.timestamp.min()[:10]+'T00:00:00Z',periods=288,freq='5min').strftime('%Y-%m-%dT%H:%M:%SZ')
    V=g.pivot(index='timestamp',columns='link_id',values='speed_kmh').reindex(index=ts,columns=links).values
    Q=g.pivot(index='timestamp',columns='link_id',values='flow_vph').reindex(index=ts,columns=links).values
    return ts,V,Q
def tinterp(x):
    x=x.copy(); t=np.arange(x.shape[0])
    for j in range(x.shape[1]):
        m=np.isfinite(x[:,j])
        if m.sum()>=2: x[:,j]=np.interp(t,t[m],x[m,j])
    return x
def common(lk_obs, tg):
    # same-time residual of observed cells vs their own time neighbours, averaged over +-3 links excluding self
    lki=tinterp(np.where(tg,np.nan,lk_obs))
    # residual of observed cell vs interpolation that excludes itself: use neighbours in time
    prev=np.vstack([np.full((1,lk_obs.shape[1]),np.nan),lki[:-1]]); nxt=np.vstack([lki[1:],np.full((1,lk_obs.shape[1]),np.nan)])
    r=np.where(tg,np.nan,lk_obs-(prev+nxt)/2)
    L=r.shape[1]; c=np.full_like(r,np.nan)
    for j in range(L):
        nb=[i for i in range(max(0,j-3),min(L,j+4)) if i!=j]
        c[:,j]=np.nanmean(r[:,nb],axis=1)
    return c,lki
f0=sorted(glob.glob(f'{R}{P}/train/mainline_states/*/*.parquet'))[0]
mp=pd.read_parquet(f0,columns=['link_id','milepost']).groupby('link_id').milepost.mean().sort_values()
links=list(mp.index)
# ---- train: optimal beta
rng=np.random.default_rng(1); X=[];Y=[]
for f in sorted(glob.glob(f'{R}{P}/train/mainline_states/*/*.parquet'))[150:175]:
    d=pd.read_parquet(f,columns=['timestamp','link_id','speed_kmh','flow_vph','is_score_eligible']); d=d[d.is_score_eligible.astype(bool)]
    ts,V,Q=arrs(d,links); lk=np.log(Q/np.maximum(V,1)); ok=np.isfinite(lk)
    tg=(rng.random(lk.shape)<0.3)&ok
    c,lki=common(np.where(ok,lk,np.nan),tg|~ok)
    m=tg&np.isfinite(c)&np.isfinite(lki)
    X.append(c[m]);Y.append(lk[m]-lki[m])
X=np.concatenate(X);Y=np.concatenate(Y)
b=np.dot(X,Y)/np.dot(X,X); print(P,'TRAIN optimal beta %.3f  corr %.3f  sd(Y) %.4f resid sd %.4f'%(b,np.corrcoef(X,Y)[0,1],Y.std(),(Y-b*X).std()))
# ---- validation: J2 deviation from interpolation vs common mode
key=pd.read_parquet('/home/user/work/tfbw/work__pub__submission_key.parquet',columns=['submission_id','task','split','panel','timestamp','station_id','link_id'])
key=key[(key.task=='state')&(key.panel==P)&(key.split=='validation')]
j2=pd.read_csv('/home/user/work/tfbw/J2_fdrecon.csv',usecols=['submission_id','task','speed_kmh','flow_vph'],dtype={'submission_id':str})
j2=j2[j2.task=='state']; key['submission_id']=key.submission_id.astype(str)
pr=key.merge(j2,on='submission_id')
X=[];Y=[]
for f in sorted(glob.glob(f'{R}{P}/validation/mainline_states_masked/*/*.parquet')):
    d=pd.read_parquet(f,columns=['timestamp','link_id','speed_kmh','flow_vph','is_score_eligible'])
    day=d.timestamp.iloc[0][:10]
    p=pr[pr.timestamp.str.startswith(day)]
    if p.empty: continue
    ts,V,Q=arrs(d[d.is_score_eligible.astype(bool)&d.speed_kmh.notna()],links)
    lk=np.log(Q/np.maximum(V,1))
    _,Vp,Qp=arrs(p,links); lkp=np.log(Qp/np.maximum(Vp,1)); tg=np.isfinite(lkp)
    c,lki=common(np.where(tg,np.nan,lk),~np.isfinite(lk)|tg)
    m=tg&np.isfinite(c)&np.isfinite(lki)
    X.append(c[m]);Y.append(lkp[m]-lki[m])
X=np.concatenate(X);Y=np.concatenate(Y)
b=np.dot(X,Y)/np.dot(X,X); print(P,'J2 implied beta %.3f corr %.3f n %d'%(b,np.corrcoef(X,Y)[0,1],len(X)))
