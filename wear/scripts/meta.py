import numpy as np, pandas as pd, pickle, sys
sys.path.insert(0,'/home/user/wear_work')
from lagshift import maj_links
Y=np.load('Y_all.npy'); I=pd.read_csv('Y_all_index.csv'); S=I.score.values; R=I.ref.astype(str).values
a=list(R).index('56944621'); A=Y[a]
d=(Y!=A).sum(1); use=np.where((d>0)&(d<=300))[0]
# probability sources
F1=np.load('kout/wear-good-fork/final_probabilities.npz')['test']; F2=np.load('kout/wear-good-fork2/final_probabilities.npz')['test']
PF=(F1+F2)/2+1e-6
st=np.load('kout/wear-good-fork/keep3/stage.npz'); WL=st['test_logp']   # window-level (pre-decode) log-probs
WL=WL-np.logaddexp.reduce(WL,axis=1,keepdims=True)
# committee support among strong files (score>=0.933, excluding today's probes)
strong=np.where((S>=0.933)&(d<=300))[0]
L1=np.load('kout/wear-good-fork/keep3/links.npz'); L2=np.load('kout/wear-good-fork2/keep3/links.npz')
succ,conf,pred,pconf=maj_links(np.vstack([L1['test_succ'],L2['test_succ']]))
def feats(w,l):
    al=A[w]
    sup=(Y[strong][:,w]==l).mean()
    nb=[x for x in (succ[w],pred[w]) if x>=0]
    nbl=np.mean([A[x]==l for x in nb]) if nb else 0.0     # neighbours already carry the new label
    nba=np.mean([A[x]==al for x in nb]) if nb else 0.0    # neighbours agree with anchor label
    return np.array([1.0,
        np.log(PF[w,l])-np.log(PF[w,al]),
        WL[w,l]-WL[w,al],
        sup,
        float(al==0 and l!=0), float(al!=0 and l==0),
        nbl, nba])
names=['const','dlogQ','dlogWin','support','null2act','act2null','nb_new','nb_anchor']
if __name__=='__main__':
    rows=[];y=[]
    for j in use:
        ws=np.where(Y[j]!=A)[0]
        rows.append(sum(feats(w,Y[j][w]) for w in ws)); y.append(S[j]-S[a])
    X=np.array(rows); y=np.array(y)
    np.save('meta_X.npy',X); np.save('meta_y.npy',y)
    print(X.shape)
