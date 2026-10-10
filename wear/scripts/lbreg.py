import numpy as np, pandas as pd
Y=np.load('Y_all.npy'); I=pd.read_csv('Y_all_index.csv'); S=I.score.values; R=I.ref.astype(str).values
a=list(R).index('56944621'); A=Y[a]
d=(Y!=A).sum(1)
use=np.where((d>0)&(d<=300))[0]
pairs={}
for j in use:
    for w in np.where(Y[j]!=A)[0]:
        pairs.setdefault((int(w),int(Y[j][w])),[]).append(j)
keys=list(pairs); print('files',len(use),'unique (w,l) pairs',len(keys),'windows',len(set(k[0] for k in keys)))
X=np.zeros((len(use),len(keys)))
pos={j:i for i,j in enumerate(use)}
for c,k in enumerate(keys):
    for j in pairs[k]: X[pos[j],c]=1
# collapse identical columns into groups
cols={}
for c in range(len(keys)):
    cols.setdefault(X[:,c].tobytes(),[]).append(c)
groups=list(cols.values()); print('groups',len(groups))
G=np.stack([X[:,g[0]] for g in groups],1)
y=S[use]-S[a]
np.save('lbreg_G.npy',G); np.save('lbreg_y.npy',y)
import pickle; pickle.dump({'keys':keys,'groups':groups,'use':use},open('lbreg.pkl','wb'))
sz=np.array([len(g) for g in groups]); nf=G.sum(0)
print('group size dist',np.percentile(sz,[50,90,99,100]),'files-per-group',np.percentile(nf,[50,90,100]))
