import numpy as np
from sklearn.metrics import f1_score
st=np.load('kout/wear-good-fork/keep3/stage.npz'); y=st['oof_y']; sb=st['oof_sbj']
P=np.load('kout/wear-good-fork/final_probabilities.npz')['oof']; p0=P.argmax(1)
def shift(p,P,sb,k):
    p=p.copy()
    for s in np.unique(sb):
        ii=np.where(sb==s)[0]
        for c in range(1,19):
            if k<0:
                jj=ii[p[ii]==c]
                if len(jj)>-k:
                    drop=jj[np.argsort(P[jj,c])[:(-k)]]; p[drop]=0
            elif k>0:
                jj=ii[p[ii]==0]
                add=jj[np.argsort(-P[jj,c])[:k]]; p[add]=c
    return p
base=f1_score(y,p0,average='macro')
print('base',round(base,5))
for k in [-6,-4,-2,-1,1,2,4]:
    q=shift(p0,P,sb,k); print(k, round(f1_score(y,q,average='macro')-base,5), 'changed',(q!=p0).sum())
# true count error stats
err=[]
for s in np.unique(sb):
    for c in range(1,19):
        err.append(((p0[sb==s]==c).sum())-((y[sb==s]==c).sum()))
err=np.array(err); print('pred-true count: mean',err.mean().round(2),'median',np.median(err),'MAE',np.abs(err).mean().round(2))
