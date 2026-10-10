import numpy as np
from sklearn.metrics import f1_score
def band_fix(p,P,sb,lo=78,hi=126,pairs=((11,12),(13,14),(16,17)),shlo=0.45,shhi=0.58,use_pairs=True):
    p=p.copy()
    for s in np.unique(sb):
        ii=np.where(sb==s)[0]
        for c in range(1,19):
            jj=ii[p[ii]==c]; n=len(jj)
            if n>hi:   # move lowest-confidence windows to their 2nd choice
                drop=jj[np.argsort(P[jj,c])[:n-hi]]
                Q=P[drop].copy(); Q[:,c]=-1; p[drop]=Q.argmax(1)
            elif n<lo:
                cand=ii[p[ii]==0]; add=cand[np.argsort(-P[cand,c])[:lo-n]]; p[add]=c
        if use_pairs:
            for a,b in pairs:   # b = complex variant
                na=(p[ii]==a).sum(); nb=(p[ii]==b).sum(); tot=na+nb
                if tot==0: continue
                sh=nb/tot
                if sh>shhi:
                    k=int(round(nb-shhi*tot)); jj=ii[p[ii]==b]; mv=jj[np.argsort(P[jj,b]-P[jj,a])[:k]]; p[mv]=a
                elif sh<shlo:
                    k=int(round(shlo*tot-nb)); jj=ii[p[ii]==a]; mv=jj[np.argsort(P[jj,a]-P[jj,b])[:k]]; p[mv]=b
    return p
if __name__=='__main__':
    st=np.load('kout/wear-good-fork/keep3/stage.npz'); y=st['oof_y']; sb=st['oof_sbj']
    for fk in ['wear-good-fork','wear-good-fork2']:
        P=np.load(f'kout/{fk}/final_probabilities.npz')['oof']; p0=P.argmax(1); b=f1_score(y,p0,average='macro')
        for lo,hi,up in [(78,126,False),(78,126,True),(70,135,True),(0,999,True),(85,120,False)]:
            q=band_fix(p0,P,sb,lo,hi,use_pairs=up)
            print(fk,lo,hi,up,round(f1_score(y,q,average='macro')-b,5),(q!=p0).sum())
    # train truth: pair shares
    for a,b2 in ((11,12),(13,14),(16,17)):
        sh=[ (y[sb==s]==b2).sum()/max(1,((y[sb==s]==a)|(y[sb==s]==b2)).sum()) for s in np.unique(sb)]
        print('true share',a,b2,np.round(np.percentile(sh,[0,5,50,95,100]),2))
