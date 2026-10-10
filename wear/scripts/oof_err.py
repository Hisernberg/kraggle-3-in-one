import numpy as np, sys
from sklearn.metrics import f1_score
st=np.load('kout/wear-good-fork/keep3/stage.npz'); y=st['oof_y']; sb=st['oof_sbj']
P=np.load(sys.argv[1])['oof']; p=P.argmax(1)
print('OOF macroF1', round(f1_score(y,p,average='macro'),4), 'acc', round((p==y).mean(),4))
# per-subject
for s in np.unique(sb):
    m=sb==s; print(s, m.sum(), round(f1_score(y[m],p[m],average='macro'),3), 'err',(p[m]!=y[m]).sum(), end=' | ')
print()
# classify errors: boundary (within 3 s of a true label change) vs interior
err=p!=y; n=len(y)
chg=np.zeros(n,bool); chg[1:]=(y[1:]!=y[:-1])|(sb[1:]!=sb[:-1])
d=np.full(n,999)
idx=np.where(chg)[0]
pos=np.arange(n); j=np.searchsorted(idx,pos)
dl=np.where(j>0, pos-idx[np.clip(j-1,0,len(idx)-1)], 999); dr=np.where(j<len(idx), idx[np.clip(j,0,len(idx)-1)]-pos, 999)
d=np.minimum(dl,dr+1)
for lo,hi in [(0,1),(1,2),(2,4),(4,10),(10,9999)]:
    m=(d>lo-1)&(d<=hi) if lo else (d<=hi)
    m=(d>=lo+1)&(d<=hi) if lo else (d<=1)
    print(f'dist {lo+1}-{hi}: windows {m.sum()} errors {err[m].sum()}')
# error type
nulli=0
t=np.where(err)[0]
print('err null->act', ((y==0)&err&(p!=0)).sum(), 'act->null', ((y!=0)&(p==0)).sum(), 'act->act', ((y!=0)&(p!=0)&err).sum())
# bout-level: contiguous runs of errors length
r=[];c=0
for e in err:
    if e: c+=1
    elif c: r.append(c); c=0
r=np.array(r); print('error runs', len(r), 'len>=10:', (r>=10).sum(), 'windows in runs>=10:', r[r>=10].sum(), 'max', r.max())
