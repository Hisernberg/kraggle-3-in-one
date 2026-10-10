"""General limb-balance bout repair: test-set windows carry one random limb, so each true bout is ~50/50
arm/leg and ~50/50 left/right. For bouts whose predicted limb split is improbable, move windows whose
link neighbours (16 Hungarian matchings) mostly sit in another bout, if the move reduces the imbalance."""
import numpy as np
from scipy.stats import binomtest
def build_nb(S):
    n=S.shape[1]; P=[[] for _ in range(n)]
    for r in range(S.shape[0]):
        for i,j in enumerate(S[r]):
            if j>=0: P[j].append(i)
    return P
def nb_labels(w,cur,S,P,depth=2):
    labs=[]; fr={w}
    for _ in range(depth):
        nx=set()
        for x in fr: nx.update(int(j) for j in S[:,x] if j>=0); nx.update(P[x])
        labs+=[cur[y] for y in nx if y!=w]; fr=nx
    return np.array(labs)
def repair(cur,sb,arm,left,S,P,pthr=0.02,share=0.36,maxmove=15,axes=('arm','left'),verbose=False):
    cur=cur.copy(); moves=[]
    for ax in axes:
        flag=arm if ax=='arm' else left
        for s in np.unique(sb):
            for c in range(1,19):
                mk=(sb==s)&(cur==c); n=int(mk.sum())
                if n<30: continue
                k=int((mk&flag).sum()); p=binomtest(k,n,.5).pvalue
                if p>=pthr: continue
                need_flag = k<n/2   # bout lacks windows of this limb type
                # candidates: windows of deficient type labelled d!=c (in same subject) whose neighbours are mostly c
                cand=np.where((sb==s)&(cur!=c)&(flag==need_flag))[0]
                sc=[]
                for w in cand:
                    L=nb_labels(w,cur,S,P)
                    if len(L)>=4: sc.append(((L==c).mean(),w))
                sc.sort(reverse=True)
                deficit=int(abs(n/2-k))
                take=[w for f,w in sc if f>=share][:min(deficit,maxmove)]
                for w in take: moves.append((w,int(cur[w]),c,ax,s,p)); cur[w]=c
                if verbose: print(ax,s,c,n,k,round(p,4),'moved',len(take))
    return cur,moves

def repair_sig(cur,sb,arm,left,S,P,pw,pthr=0.05,share=0.36,sig=0.6,maxmove=15,axes=('arm','left'),verbose=False):
    """Like repair(), but only moves windows from class d into bout c when the bout's own windows of the
    deficient limb type are mostly classified as d by the window model (pw) -> c and d are confusable on that limb."""
    cur=cur.copy(); moves=[]
    for ax in axes:
        flag=arm if ax=='arm' else left
        for s in np.unique(sb):
            for c in range(1,19):
                mk=(sb==s)&(cur==c); n=int(mk.sum())
                if n<30: continue
                k=int((mk&flag).sum()); p=binomtest(k,n,.5).pvalue
                if p>=pthr: continue
                need_flag = k<n/2
                own=np.where(mk&(flag==need_flag))[0]
                if len(own)<5: continue
                vals,cnt=np.unique(pw[own],return_counts=True)
                d=int(vals[np.argmax(cnt)]); fr=cnt.max()/len(own)
                if d==c or fr<sig: continue
                cand=np.where((sb==s)&(cur==d)&(flag==need_flag))[0]
                sc=[]
                for w in cand:
                    L=nb_labels(w,cur,S,P)
                    if len(L)>=4: sc.append(((L==c).mean(),w))
                sc.sort(reverse=True)
                deficit=int(abs(n/2-k))
                take=[w for f,w in sc if f>=share][:min(deficit,maxmove)]
                for w in take: moves.append((w,int(cur[w]),c,ax,s,p)); cur[w]=c
                if verbose: print(ax,s,c,'n',n,'k',k,'p',round(p,4),'confused-with',d,round(fr,2),'moved',len(take))
    return cur,moves
