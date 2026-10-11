import numpy as np, pandas as pd, lightgbm as lgb
D='/home/user/wear_data/train/inertial_feat/'
def tiles(s,limb):
    d=pd.read_csv(D+s+'.csv',low_memory=False); X=d[[f'{limb}_acc_{a}' for a in 'xyz']].values.astype(np.float32)
    n=len(X)//50; return X[:n*50].reshape(n,50,3)
def cands(T,K=30):
    n=len(T); last=T[:,-4:,:]; slope=(last[:,-1]-last[:,0])/3
    P=(last[:,-1]+slope).reshape(n,-1); F=T[:,0,:].reshape(n,-1)
    C=((P[:,None]-F[None])**2).sum(-1); np.fill_diagonal(C,1e9)
    return np.argsort(C,1)[:,:K],C
def feats(T,i,j):
    a=T[i]; b=T[j]
    f=[]
    f+=list(b[0]-a[-1]); f+=list((b[1]-b[0])-(a[-1]-a[-2]))           # value / slope jump
    f+=list(b[:5].mean(0)-a[-5:].mean(0)); f+=list(b[:10].mean(0)-a[-10:].mean(0))
    f+=list(b.std(0)-a.std(0)); f+=list(b.mean(0)-a.mean(0))
    f+=[np.linalg.norm(b[0]-a[-1])/(np.std(np.diff(a,axis=0))+1e-3)]
    f+=[np.corrcoef(np.linalg.norm(a,axis=1)[-25:],np.linalg.norm(b,axis=1)[:25])[0,1] if True else 0]
    return f
def build(subjects,limbs):
    X=[];Y=[];G=[];meta=[]
    for s in subjects:
        for limb in limbs:
            T=tiles(s,limb); n=len(T); cand,C=cands(T)
            for i in range(n-1):
                cs=list(cand[i])
                if i+1 not in cs: cs[-1]=i+1     # make sure the true successor is a candidate (recall measured separately)
                for j in cs:
                    X.append(feats(T,i,j)+[C[i,j],cs.index(j)]); Y.append(int(j==i+1)); G.append((s,limb,i))
    return np.nan_to_num(np.array(X,np.float32)),np.array(Y),G
tr=['sbj_3','sbj_8','sbj_11','sbj_16']; te=['sbj_5','sbj_19']
limbs=['right_arm','left_leg']
Xtr,Ytr,_=build(tr,limbs); print('train rows',len(Ytr),flush=True)
m=lgb.LGBMClassifier(n_estimators=300,learning_rate=0.05,num_leaves=63,verbose=-1).fit(Xtr,Ytr)
for s in te:
    for limb in limbs:
        T=tiles(s,limb); n=len(T); cand,C=cands(T)
        rec=np.mean([i+1 in cand[i] for i in range(n-1)])
        hit=0
        for i in range(n-1):
            cs=list(cand[i]); F=np.nan_to_num(np.array([feats(T,i,j)+[C[i,j],k] for k,j in enumerate(cs)],np.float32))
            hit+= cs[int(np.argmax(m.predict_proba(F)[:,1]))]==i+1
        print(s,limb,'candidate recall@30',round(rec,3),'learned top1 exact',round(hit/(n-1),3),flush=True)
