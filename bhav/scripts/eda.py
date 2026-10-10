import soundfile as sf, pandas as pd, numpy as np, hashlib, sys
D='/home/user/kraggle-3-in-one/bhav/work/data/SRCASW-BhavVaani'
tr=pd.read_csv(f'{D}/train.csv'); te=pd.read_csv(f'{D}/test.csv')
print(tr.emotion.value_counts())
rows=[]
for split,df in [('train',tr),('test',te)]:
    for _,r in df.iterrows():
        p=f'{D}/{split}/{r.filename}'
        info=sf.info(p); x,sr=sf.read(p,dtype='int16',always_2d=True)
        rows.append(dict(split=split,Id=r.Id,emotion=r.get('emotion',None),sr=sr,ch=info.channels,sub=info.subtype,fmt=info.format,
          n=len(x),dur=len(x)/sr,peak=int(np.abs(x).max()),rms=float(np.sqrt((x.astype(float)**2).mean())),
          md5=hashlib.md5(x.tobytes()).hexdigest(), fsize=__import__('os').path.getsize(p),
          dc=float(x.mean()), lead0=int(np.argmax(np.abs(x[:,0])>50)) ))
m=pd.DataFrame(rows); m.to_csv('meta.csv',index=False)
print(m.groupby('split')[['sr','ch']].agg(lambda s: dict(s.value_counts())))
print(m.groupby(['split','sub']).size())
print(m.groupby('split').dur.describe())
print(m[m.split=='train'].groupby('emotion')[['dur','rms','peak']].median())
print(pd.crosstab(m[m.split=='train'].emotion, m[m.split=='train'].sr))
print(pd.crosstab(m[m.split=='train'].emotion, m[m.split=='train'].ch))
d=m[m.md5.duplicated(keep=False)].sort_values('md5'); print('exact dup rows',len(d)); print(d[['split','Id','emotion','md5']].head(40))
# Id vs emotion pattern
t=m[m.split=='train'].sort_values('Id'); print(''.join(t.emotion.str[0].tolist())[:400])
