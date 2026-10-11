"""Do competition clips come from public datasets? Mel-fingerprint match of every external clip."""
import glob, os, pickle, sys, re
import numpy as np, pandas as pd, librosa
from joblib import Parallel, delayed
sys.path.insert(0, os.path.dirname(__file__))
from common import BH, load_meta
X = f'{BH}/work/ext'
files = sorted(glob.glob(f'{X}/**/*.wav', recursive=True))

def mel(p):
    try:
        x, sr = librosa.load(p, sr=16000)
    except Exception:
        return None
    lm = np.log(librosa.feature.melspectrogram(y=x, sr=16000, n_fft=512, hop_length=160, n_mels=40) + 1e-8)
    return ((lm - lm.mean()) / (lm.std() + 1e-6)).astype(np.float32), len(x)

E = Parallel(n_jobs=4)(delayed(mel)(p) for p in files)
keep = [i for i, e in enumerate(E) if e is not None]
files = [files[i] for i in keep]; E = [E[i] for i in keep]
pickle.dump((files, E), open(f'{BH}/work/ext_mels.pkl', 'wb'))
F = pickle.load(open(sys.argv[1], 'rb')); m = load_meta()
emb = lambda f: np.r_[f.mean(1), f.std(1)]
A = np.array([emb(f) for f in F]); B = np.array([emb(e[0]) for e in E])
mu, sd = A.mean(0), A.std(0); A = (A - mu) / sd; B = (B - mu) / sd
A /= np.linalg.norm(A, axis=1, keepdims=True); B /= np.linalg.norm(B, axis=1, keepdims=True)
S = A @ B.T; nn = np.argsort(-S, 1)[:, :8]

def xc(a, b):
    if a.shape[1] > b.shape[1]: a, b = b, a
    L = a.shape[1]; return max(np.corrcoef(a.ravel(), b[:, o:o + L].ravel())[0, 1] for o in range(0, b.shape[1] - L + 1))
res = Parallel(n_jobs=4)(delayed(lambda i: [(i, j, xc(F[i], E[j][0])) for j in nn[i]])(i) for i in range(len(F)))
R = pd.DataFrame([r for rr in res for r in rr], columns=['i', 'j', 'xc'])
R = R.sort_values('xc', ascending=False).drop_duplicates('i')
R['split'] = m.split.values[R.i]; R['Id'] = m.Id.values[R.i]; R['emotion'] = m.emotion.values[R.i]
R['ext'] = [files[j].replace(X + '/', '') for j in R.j]
R.to_csv(f'{BH}/work/ext_match.csv', index=False)
print(np.histogram(R.xc, bins=[0, .6, .7, .8, .85, .9, .95, .99, 1.01]))
print(R.head(25).to_string())
