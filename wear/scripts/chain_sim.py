"""Simulate 2nd-challenge-style complete per-limb streams from train recordings and measure how well
1-s tiles can be re-chained by sample continuity (with optional organiser-style augmentation)."""
import numpy as np, pandas as pd, sys
from scipy.optimize import linear_sum_assignment
rng=np.random.default_rng(0)
def tiles(f,limb):
    d=pd.read_csv(f,low_memory=False); cols=[f'{limb}_acc_{a}' for a in 'xyz']
    X=d[cols].values.astype(np.float32); n=len(X)//50
    return X[:n*50].reshape(n,50,3)
def augment(T,frac):
    T=T.copy(); k=rng.random(len(T))<frac
    T[k]=T[k]*rng.normal(1,0.1,(k.sum(),1,3))+rng.normal(0,0.05,T[k].shape)   # scaling + jitter
    return T
def chain_acc(T):
    n=len(T)
    # predict the first 3 samples of the next tile by linear extrapolation of the last 5
    last=T[:,-5:,:]; slope=(last[:,-1]-last[:,0])/4
    pred=np.stack([last[:,-1]+slope*(k+1) for k in range(3)],1)      # n,3,3
    first=T[:,:3,:]
    # cost[i,j] = distance between pred of i and first of j  (chunked)
    C=np.empty((n,n),np.float32)
    P=pred.reshape(n,-1); F=first.reshape(n,-1)
    for s in range(0,n,1000):
        C[s:s+1000]=((P[s:s+1000,None,:]-F[None,:,:])**2).sum(-1)
    np.fill_diagonal(C,1e9)
    top1=C.argmin(1); acc1=(top1[:-1]==np.arange(1,n)).mean()
    r,c=linear_sum_assignment(C[:-1])   # one-to-one assignment (last tile has no successor)
    hung=(c==np.arange(1,n)).mean()
    return round(float(acc1),4),round(float(hung),4)
for f in sys.argv[1:]:
    for limb in ['right_arm','left_leg']:
        T=tiles(f,limb)
        print(f.split('/')[-1],limb,len(T),'clean top1/hungarian',chain_acc(T),'| 30% augmented',chain_acc(augment(T,0.3)),flush=True)
