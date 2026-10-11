import numpy as np, pandas as pd, sys
from scipy.optimize import linear_sum_assignment
def load(f):
    d=pd.read_csv(f,low_memory=False); lab=d['label'].fillna('null').values
    return d,lab
def run(f):
    d,lab=load(f)
    for limb in ['right_arm','left_leg']:
        X=d[[f'{limb}_acc_{a}' for a in 'xyz']].values.astype(np.float64); n=len(X)//50
        T=X[:n*50].reshape(n,50,3); y=lab[25::50][:n]
        # smoothness cost: fit quadratic to last 8 samples of i, extrapolate 4 samples; compare with first 4 of j (scaled by local noise)
        t=np.arange(8)
        A=np.vstack([t**2,t,np.ones(8)]).T; pinv=np.linalg.pinv(A)
        tail=T[:,-8:,:]                                     # n,8,3
        coef=np.einsum('kt,ntc->nkc',pinv,tail)             # n,3,3
        tf=np.arange(8,12); Af=np.vstack([tf**2,tf,np.ones(4)]).T
        pred=np.einsum('tk,nkc->ntc',Af,coef)               # n,4,3
        noise=np.std(np.diff(T,axis=1),axis=1).mean(1)+1e-3 # n
        P=pred.reshape(n,-1); F=T[:,:4,:].reshape(n,-1)
        C=np.empty((n,n))
        for s in range(0,n,800): C[s:s+800]=((P[s:s+800,None]-F[None])**2).sum(-1)/noise[s:s+800,None]**2
        np.fill_diagonal(C,1e12)
        r,c=linear_sum_assignment(C[:-1]); succ=np.full(n,-1); succ[r]=c
        ex=(succ[:-1]==np.arange(1,n)).mean(); same=(y[succ[:-1]]==y[1:n][:len(succ)-1] if False else (y[succ[:-1]]==y[:-1])).mean()
        print(f.split('/')[-1],limb,'exact',round(ex,3),'same-label',round(same,3),'(true successor same-label',round((y[1:]==y[:-1]).mean(),3),')',flush=True)
for f in sys.argv[1:]: run(f)
