import sys; sys.path.insert(0,'/home/user/wear_work')
import numpy as np, pandas as pd
import meta
m=pd.read_csv('/home/user/wear_data/test/test_meta_data.csv'); sb=m.sbj_id.values; loc=m.sensor_location.values
armm=np.isin(loc,['left_arm','right_arm'])
L1=np.load('kout/wear-good-fork/keep3/links.npz'); L2=np.load('kout/wear-good-fork2/keep3/links.npz')
S=np.vstack([L1['test_succ'],L2['test_succ']]); n=S.shape[1]
P=[[] for _ in range(n)]
for r in range(S.shape[0]):
    for i,j in enumerate(S[r]):
        if j>=0: P[j].append(i)
def nb_labels(w,cur,depth=2):
    labs=[]; fr={w}
    for _ in range(depth):
        nx=set()
        for x in fr: nx.update(int(j) for j in S[:,x] if j>=0); nx.update(P[x])
        labs+=[cur[y] for y in nx if y!=w]; fr=nx
    return np.array(labs)
def inspect(cur,s,c,arm_side):
    mk=(sb==s)&(cur==c)&(armm if arm_side else ~armm)
    rows=[]
    for w in np.where(mk)[0]:
        L=nb_labels(w,cur)
        if len(L)==0: continue
        vals,cnt=np.unique(L,return_counts=True); fr=dict(zip(vals,cnt/len(L)))
        own=fr.get(c,0); other=max(((v,f) for v,f in fr.items() if v!=c),key=lambda t:t[1],default=(None,0))
        rows.append((w,loc[w],round(own,2),other[0],round(other[1],2),len(L),round(meta.PF[w,c],2),round(meta.PF[w,other[0]],2) if other[0] is not None else 0))
    return pd.DataFrame(rows,columns=['w','loc','own','alt','alt_frac','nnb','Qown','Qalt']).sort_values('own')
if __name__=='__main__':
    cur=pd.read_csv(sys.argv[1]).target_feature.values
    for s,c,arm in [(24,10,True),(23,12,False)]:
        print('== sbj',s,'class',c,'arm' if arm else 'leg'); print(inspect(cur,s,c,arm).head(25).to_string(index=False))
