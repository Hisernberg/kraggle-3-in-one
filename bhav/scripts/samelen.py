import pandas as pd, numpy as np, pickle, soundfile as sf, itertools
D='/home/user/kraggle-3-in-one/bhav/work/data/SRCASW-BhavVaani'
m=pd.read_csv('meta.csv'); F=pickle.load(open('mels.pkl','rb'))
m['filename']=m.Id.map(lambda i:f'audio_{i:06d}.wav')
W={}
def wav(i):
    if i not in W:
        x,_=sf.read(f'{D}/{m.split[i]}/{m.filename[i]}',dtype='float32'); W[i]=x
    return W[i]
rows=[]
for n,g in m.groupby('n'):
    idx=g.index.tolist()
    for i,j in itertools.combinations(idx,2):
        a,b=F[i],F[j]; mc=np.corrcoef(a.ravel(),b.ravel())[0,1]
        wa,wb=wav(i),wav(j); wc=np.corrcoef(wa,wb)[0,1]
        rows.append((i,j,n,mc,wc))
R=pd.DataFrame(rows,columns=['i','j','n','melc','wavc'])
R['s1']=m.split[R.i].values; R['s2']=m.split[R.j].values; R['e1']=m.emotion[R.i].values; R['e2']=m.emotion[R.j].values
R['id1']=m.Id[R.i].values; R['id2']=m.Id[R.j].values
R.to_csv('samelen_pairs.csv',index=False)
print(len(R)); print(np.histogram(R.melc,bins=[-1,0,.3,.5,.6,.7,.8,.85,.9,.95,.99,1.01]))
print(np.histogram(R.wavc,bins=[-1,0,.1,.3,.5,.7,.9,.99,1.01]))
tt=R[(R.s1=='train')&(R.s2=='train')]
for th in [0.5,0.6,0.7,0.8,0.85,0.9]:
    s=tt[tt.melc>th]; print(th,len(s),'same label',(s.e1==s.e2).mean().round(3))
