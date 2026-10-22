"""K1: J2 + same-time spatial common-mode correction of log flow at free-flow Task 1 target cells.
beta_opt (train, simulated masks with the release regime mix) vs beta_J2 (J2's own slope on the same signal);
apply q *= exp(shrink*(beta_opt-beta_J2)*c) where J2 speed >= 0.8 vf."""
import glob, numpy as np, pandas as pd, sys, warnings; warnings.filterwarnings('ignore')
R='/home/user/data/kaggle_public/corridors/'
SHRINK=float(sys.argv[1]) if len(sys.argv)>1 else 0.5
OUT=sys.argv[2] if len(sys.argv)>2 else '/home/user/work/subs/K1_cm.csv'
PANELS=sorted(p.split('/')[-1] for p in glob.glob(R+'*'))
def tinterp(x):
    x=x.copy(); t=np.arange(x.shape[0])
    for j in range(x.shape[1]):
        m=np.isfinite(x[:,j])
        if m.sum()>=2: x[:,j]=np.interp(t,t[m],x[m,j])
    return x
def common(lobs):  # lobs: observed log q with NaN at unobserved
    lki=tinterp(lobs); L=lobs.shape[1]
    prev=np.vstack([np.full((1,L),np.nan),lki[:-1]]); nxt=np.vstack([lki[1:],np.full((1,L),np.nan)])
    r=lobs-(prev+nxt)/2
    c=np.full_like(r,np.nan)
    for j in range(L):
        nb=[i for i in range(max(0,j-3),min(L,j+4)) if i!=j]
        c[:,j]=np.nanmean(r[:,nb],axis=1)
    return np.clip(c,-0.3,0.3),lki
def grid(df,links,day,cols):
    ts=pd.date_range(day+'T00:00:00Z',periods=288,freq='5min').strftime('%Y-%m-%dT%H:%M:%SZ')
    g=df.groupby(['timestamp','link_id'],as_index=False)[cols].mean()
    return ts,[g.pivot(index='timestamp',columns='link_id',values=c).reindex(index=ts,columns=links).values for c in cols]
key=pd.read_parquet('/home/user/work/tfbw/work__pub__submission_key.parquet',columns=['submission_id','task','split','panel','timestamp','link_id'])
key=key[key.task=='state'].drop(columns='task'); key['submission_id']=key.submission_id.astype(np.int64)
sub=pd.read_csv('/home/user/work/tfbw/J2_fdrecon.csv',dtype={'queue_pred':str,'path_flow':str})
st=sub[sub.task=='state'][['submission_id','speed_kmh','flow_vph']].merge(key,on='submission_id')
newq=pd.Series(dtype=float)
rng=np.random.default_rng(7); log=[]
for P in PANELS:
    fd=pd.read_csv(f'{R}{P}/network/fd_parameters.csv'); vf=fd.groupby('link_id').free_speed_kmh.mean()
    f0=sorted(glob.glob(f'{R}{P}/train/mainline_states/*/*.parquet'))[0]
    links=list(pd.read_parquet(f0,columns=['link_id','milepost']).groupby('link_id').milepost.mean().sort_values().index)
    vfa=vf.reindex(links).fillna(105).values[None,:]
    # train optimal beta (free-flow cells), regime mix 0.2/0.3/0.5
    X=[];Y=[]
    for f in sorted(glob.glob(f'{R}{P}/train/mainline_states/*/*.parquet'))[120:160]:
        d=pd.read_parquet(f,columns=['timestamp','link_id','speed_kmh','flow_vph','is_score_eligible']); d=d[d.is_score_eligible.astype(bool)]
        _,(V,Q)=grid(d,links,d.timestamp.iloc[0][:10],['speed_kmh','flow_vph'])
        lq=np.log(np.maximum(Q,1)); ok=np.isfinite(lq)
        tg=(rng.random(lq.shape)<rng.choice([0.2,0.3,0.5]))&ok
        c,lki=common(np.where(ok&~tg,lq,np.nan))
        m=tg&np.isfinite(c)&np.isfinite(lki)&(V>=0.8*vfa)
        X.append(c[m]);Y.append(lq[m]-lki[m])
    X=np.concatenate(X);Y=np.concatenate(Y); bopt=X@Y/(X@X)
    # J2 slope and correction, both splits
    sp=st[st.panel==P]; X=[];Y=[];upd=[]
    for split in ('validation','private'):
        for f in sorted(glob.glob(f'{R}{P}/{split}/mainline_states_masked/*/*.parquet')):
            d=pd.read_parquet(f,columns=['timestamp','link_id','speed_kmh','flow_vph','is_score_eligible'])
            day=d.timestamp.iloc[0][:10]
            p=sp[(sp.split==split)&sp.timestamp.str.startswith(day)]
            if p.empty: continue
            ts,(V,Q)=grid(d[d.is_score_eligible.astype(bool)&d.flow_vph.notna()],links,day,['speed_kmh','flow_vph'])
            c,lki=common(np.log(np.maximum(Q,1)))
            ti={t:i for i,t in enumerate(ts)}; li={l:j for j,l in enumerate(links)}
            ii=p.timestamp.map(ti).values; jj=p.link_id.map(li).values
            okk=~(pd.isna(ii)|pd.isna(jj)); ii=ii[okk].astype(int); jj=jj[okk].astype(int); pp=p[okk]
            cc=c[ii,jj]; base=lki[ii,jj]; free=pp.speed_kmh.values>=0.8*vfa[0,jj]
            m=np.isfinite(cc)&np.isfinite(base)&free
            if split=='validation': X.append(cc[m]);Y.append(np.log(np.maximum(pp.flow_vph.values[m],1))-base[m])
            upd.append(pd.DataFrame({'submission_id':pp.submission_id.values[m],'c':cc[m]}))
    X=np.concatenate(X);Y=np.concatenate(Y); bj2=X@Y/(X@X)
    db=SHRINK*max(0.0,bopt-bj2)
    u=pd.concat(upd); u['fac']=np.exp(db*u.c)
    log.append((P,round(bopt,3),round(bj2,3),round(db,3),len(u),round(float(np.abs(u.fac-1).mean()),5)))
    print(log[-1],flush=True)
    newq=pd.concat([newq,u.set_index('submission_id').fac])
sub=sub.set_index('submission_id')
idx=newq.index
sub.loc[idx,'flow_vph']=(sub.loc[idx,'flow_vph'].astype(float)*newq.values).round(4)
sub.reset_index().to_csv(OUT,index=False)
print('wrote',OUT,'rows changed',len(idx))
