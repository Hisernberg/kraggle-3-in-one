import sys; sys.path.insert(0,'/home/user/wear_work')
import numpy as np, pandas as pd, itertools
import meta
K='kout/'
def nl(p): p=np.asarray(p,np.float64)+1e-6; return np.log(p/p.sum(1,keepdims=True))
def lsm(l): l=np.asarray(l,np.float64); return l-np.logaddexp.reduce(l,axis=1,keepdims=True)
SRC={
 'fus': nl(np.mean([np.load(K+f'wear-fusion-full/test_full_s{i}.npy') for i in range(3)],0)),
 'uec': nl((np.load(K+'wear-uec-k1/test_cnn8.npy')+np.load(K+'wear-uec-k1/test_xcepvid.npy'))/2),
 'hbP': nl(np.load(K+'wear-hanbat-gpu/keep/P_test.npy')),
 'imu': lsm(np.mean([np.load(K+f'wear-hanbat-gpu/keep/test_logp_raw_imu_s{i}.npy') for i in range(2)],0)),
}
def extra(w,l,a): return np.array([SRC[k][w,l]-SRC[k][w,a] for k in SRC])
ENAMES=list(SRC)
def build(anchor, files_mask_idx, files_y):
    rows=[]
    for j in files_mask_idx:
        ws=np.where(meta.Y[j]!=anchor)[0]
        rows.append(sum(np.r_[meta.feats(w,meta.Y[j][w]),extra(w,meta.Y[j][w],anchor[w])] for w in ws))
    return np.array(rows)
if __name__=='__main__':
    A=meta.A; use=meta.use
    X=build(A,use,None); y=meta.S[use]-meta.S[list(meta.R).index('56944621')]
    m6=pd.read_csv('cands/M6_meta_top6.csv').target_feature.values
    ws=np.where(m6!=A)[0]; X=np.vstack([X,sum(np.r_[meta.feats(w,m6[w]),extra(w,m6[w],A[w])] for w in ws)]); y=np.r_[y,0.00043]
    np.save('meta3_X.npy',X); np.save('meta3_y.npy',y)
    names=meta.names+ENAMES; n=len(y)
    def loo(cols):
        e=[]
        for i in range(n):
            m=np.arange(n)!=i; b=np.linalg.lstsq(X[m][:,cols],y[m],rcond=None)[0]; e.append((X[i,cols]@b-y[i])**2)
        return np.sqrt(np.mean(e))
    res=sorted((loo([0,*c]),[names[k] for k in [0,*c]]) for k in range(1,5) for c in itertools.combinations(range(1,len(names)),k))
    for r in res[:10]: print(round(r[0],6),r[1])
