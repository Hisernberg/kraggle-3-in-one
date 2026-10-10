import numpy as np
from collections import Counter
def maj_links(succ_mat):
    n=succ_mat.shape[1]; succ=np.full(n,-1); conf=np.zeros(n)
    for i in range(n):
        c=Counter(int(x) for x in succ_mat[:,i] if x>=0)
        if c: j,k=c.most_common(1)[0]; succ[i]=j; conf[i]=k/succ_mat.shape[0]
    pred=np.full(n,-1); pconf=np.zeros(n)
    for i in np.argsort(conf):   # higher conf overwrite
        if succ[i]>=0: pred[succ[i]]=i; pconf[succ[i]]=conf[i]
    return succ,conf,pred,pconf
def lag_shift(p,succ,conf,pred,pconf,direction=+1,minconf=0.75,do_start=True,do_end=True):
    """direction +1: labels later than sensor -> trim set start, extend set end by one window"""
    q=p.copy(); n=len(p); ch=[]
    for i in range(n):
        c=p[i]
        if c==0: continue
        j=pred[i]
        if do_start and j>=0 and pconf[i]>=minconf and p[j]==0:      # i is set start
            if direction>0: ch.append((i,0))
            else: ch.append((j,c))
        k=succ[i]
        if do_end and k>=0 and conf[i]>=minconf and p[k]==0:          # i is set end
            if direction>0: ch.append((k,c))
            else: ch.append((i,0))
    for w,v in ch: q[w]=v
    return q,len(ch)
if __name__=='__main__':
    from sklearn.metrics import f1_score
    st=np.load('kout/wear-good-fork/keep3/stage.npz'); y=st['oof_y']; sb=st['oof_sbj']
    L=np.load('kout/wear-good-fork/keep3/links.npz')
    succ,conf,pred,pconf=maj_links(L['oof_succ'])
    true_succ=np.r_[np.arange(1,len(y)),-1]; same=np.r_[sb[1:]==sb[:-1],False]
    print('maj succ exact',np.mean(succ[same]==true_succ[same]).round(3))
    for f in ['wear-good-fork']:
        p=np.load(f'kout/{f}/final_probabilities.npz')['oof'].argmax(1); b=f1_score(y,p,average='macro')
        for d in [+1,-1]:
            for mc in [0.5,0.75,1.0]:
                for ds,de in [(1,1),(1,0),(0,1)]:
                    q,nc=lag_shift(p,succ,conf,pred,pconf,d,mc,ds,de)
                    print(f,'dir',d,'minconf',mc,'start',ds,'end',de,'changes',nc,'dF1',round(f1_score(y,q,average='macro')-b,5))
