import numpy as np, pandas as pd, sys
from scipy.optimize import linear_sum_assignment
for f in sys.argv[1:]:
    d=pd.read_csv(f,low_memory=False); lab=d['label'].fillna('null').values
    def T(limb):
        X=d[[f'{limb}_acc_{a}' for a in 'xyz']].values.astype(np.float64); n=len(X)//50
        return X[:n*50].reshape(n,50,3)
    A=T('right_arm'); B=T('left_leg'); n=len(A); y=lab[25::50][:n]
    def env(W):
        m=np.linalg.norm(W,axis=2); m=m-m.mean(1,keepdims=True); return m/(np.linalg.norm(m,axis=1,keepdims=True)+1e-9)
    ea,eb=env(A),env(B)
    S=ea@eb.T                                     # correlation of magnitude envelopes (same timing => high)
    # allow small lags (+-3 samples) via max over shifted correlations
    best=S.copy()
    for lag in [-3,-2,-1,1,2,3]:
        eb2=np.roll(eb,lag,axis=1); best=np.maximum(best,ea@eb2.T)
    top1=(best.argmax(1)==np.arange(n)).mean()
    r,c=linear_sum_assignment(-best); hung=(c==np.arange(n)).mean()
    act=y!='null'
    print(f.split('/')[-1],'n',n,'same-second match top1',round(top1,3),'hungarian',round(hung,3),'| active windows top1',round((best.argmax(1)==np.arange(n))[act].mean(),3),flush=True)
