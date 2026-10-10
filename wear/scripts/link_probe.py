import numpy as np, sys
D='/home/user/wear_data/train/videomae_feat/'
def probe(s):
    v=np.load(D+s+'.npy').astype(np.float32)
    n=v.shape[0]//30
    w=v[:n*30].reshape(n,30,768)[:,8:23]          # 15 central frames
    def nz(a): return a/np.linalg.norm(a,axis=-1,keepdims=True)
    last=w[:,-1]; first=w[:,0]
    ext=w[:,-1]+(w[:,-1]-w[:,-4])/3*16/1.0*0.5     # crude linear extrapolation
    res={}
    for name,a,b in [('last-first',last,first),('mean-mean',w.mean(1),w.mean(1)),('ext-first',ext,first),('last3-first3',w[:,-3:].mean(1),w[:,:3].mean(1))]:
        S=nz(a)@nz(b).T; np.fill_diagonal(S,-9)
        top=S.argmax(1); acc=(top[:-1]==np.arange(1,n)).mean()
        res[name]=round(acc,4)
    # true successor rank under last-first
    S=nz(last)@nz(first).T; np.fill_diagonal(S,-9)
    r=(S[np.arange(n-1)]>S[np.arange(n-1),np.arange(1,n)][:,None]).sum(1)
    res['rank<=5']=round((r<5).mean(),4)
    print(s,n,res,flush=True)
for s in sys.argv[1:]: probe(s)
