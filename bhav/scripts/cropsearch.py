"""Exhaustive search for clips that are crops (sub-segments) of other clips.

Signature: first 4 PCA components of the per-clip-normalised log-mel frames. For every pair
(short a, long b) the Pearson correlation of a with each window of b is computed by FFT; the
best offset is then verified with the full 40-band log-mel correlation.
"""
import pickle, sys, os
import numpy as np, pandas as pd
from joblib import Parallel, delayed
sys.path.insert(0, os.path.dirname(__file__))
from common import BH, load_meta

S = sys.argv[1]   # scratch dir with mels.pkl
F = pickle.load(open(f'{S}/mels.pkl', 'rb')); m = load_meta()
allf = np.concatenate([f.T for f in F])
mu = allf.mean(0); U = np.linalg.svd(allf[::7] - mu, full_matrices=False)[2][:4]
sig = [((f.T - mu) @ U.T).T for f in F]                          # 4 x T
sig = [(s - s.mean(1, keepdims=True)) / (s.std(1, keepdims=True) + 1e-6) for s in sig]
L = np.array([s.shape[1] for s in sig]); N = len(sig)


def for_long(b):
    B = sig[b]; Lb = L[b]; nfft = 1 << int(np.ceil(np.log2(2 * Lb)))
    FB = np.fft.rfft(B, nfft)
    c1 = np.concatenate([np.zeros((4, 1)), np.cumsum(B, 1)], 1); c2 = np.concatenate([np.zeros((4, 1)), np.cumsum(B ** 2, 1)], 1)
    out = []
    for a in np.where((L <= Lb) & (L >= 40) & (np.arange(N) != b))[0]:
        if L[a] == Lb and a > b:
            continue
        A = sig[a]; La = L[a]
        xc = np.fft.irfft(np.conj(np.fft.rfft(A, nfft)) * FB, nfft)[:, :Lb - La + 1]   # sum_t a_t b_{o+t}
        s1 = c1[:, La:] - c1[:, :-La] if La < Lb + 1 else None
        s1 = (c1[:, La:Lb + 1] - c1[:, :Lb - La + 1]); s2 = (c2[:, La:Lb + 1] - c2[:, :Lb - La + 1])
        sd = np.sqrt(np.maximum(s2 / La - (s1 / La) ** 2, 1e-8))
        r = (xc / (La * sd)).mean(0); o = int(r.argmax())
        if r[o] > 0.8:
            out.append((a, b, o, float(r[o])))
    return out


res = Parallel(n_jobs=4, batch_size=8)(delayed(for_long)(b) for b in range(N))
rows = [x for r in res for x in r]
R = pd.DataFrame(rows, columns=['a', 'b', 'off', 'sig_r'])
R['mel_r'] = [np.corrcoef(F[a].ravel(), F[b][:, o:o + F[a].shape[1]].ravel())[0, 1] for a, b, o in zip(R.a, R.b, R.off)]
for k in ['split', 'Id', 'emotion']:
    R[k + '_a'] = m[k].values[R.a]; R[k + '_b'] = m[k].values[R.b]
R['La'] = L[R.a]; R['Lb'] = L[R.b]
R.to_csv(f'{BH}/work/crop_pairs.csv', index=False)
tt = R[(R.split_a == 'train') & (R.split_b == 'train')]
for th in [0.8, 0.85, 0.9, 0.95]:
    s = tt[tt.mel_r > th]; print(th, len(s), 'same', round((s.emotion_a == s.emotion_b).mean(), 3),
                                 'crops(La<Lb-5)', (s.La < s.Lb - 5).sum(), 'test-involved', ((R.mel_r > th) & ((R.split_a == 'test') | (R.split_b == 'test'))).sum())
