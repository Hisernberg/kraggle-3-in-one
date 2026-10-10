import pandas as pd, numpy as np
m=pd.read_csv('meta.csv')
R=pd.read_csv('samelen_pairs.csv'); P=pd.read_csv('pairs.csv')
tt=R[(R.s1=='train')&(R.s2=='train')]
for nm,s in [('wav>.9',tt[tt.wavc>.9]),('mel>.85&wav<.9',tt[(tt.melc>.85)&(tt.wavc<.9)])]:
    print(nm,len(s),(s.e1==s.e2).mean())
E=R[(R.wavc>0.9)|(R.melc>0.85)][['i','j','melc','wavc']]
# union find clusters
par=list(range(len(m)))
def f(x):
    while par[x]!=x: par[x]=par[par[x]]; x=par[x]
    return x
for i,j in zip(E.i,E.j): par[f(i)]=f(j)
m['cl']=[f(i) for i in range(len(m))]
cs=m.groupby('cl').size(); m['clsize']=m.cl.map(cs)
print('files in clusters >1:', (m.clsize>1).sum(), 'train',((m.clsize>1)&(m.split=='train')).sum(),'test',((m.clsize>1)&(m.split=='test')).sum())
# test with train mate
lab={}
for c,g in m.groupby('cl'):
    l=g[g.split=='train'].emotion.dropna()
    lab[c]=l.value_counts().to_dict()
m['mate']=m.cl.map(lab)
te=m[m.split=='test']
has=te[te.mate.map(len)>0]; print('test with train mate:',len(has),'of',len(te))
conf=has[has.mate.map(len)>1]; print('conflicting mates',len(conf)); print(conf[['Id','mate']])
print('test mate label dist', has.mate.map(lambda d:max(d,key=d.get)).value_counts().to_dict())
# conflicts within train clusters
for c,g in m[m.split=='train'].groupby('cl'):
    if g.emotion.nunique()>1: print('train conflict cluster',g[['Id','emotion']].values.tolist())
print('cluster size dist',m[m.clsize>1].groupby('cl').size().value_counts().to_dict())
m.to_csv('meta_cl.csv',index=False)
