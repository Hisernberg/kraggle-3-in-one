"""Layer-wise logistic-regression probes over pooled SSL embeddings.

For every model and layer: 5-fold stratified CV (seed 0) on train, macro-F1 of the model alone on
'novel' rows (no duplicate in the training folds) and on all rows. Writes work/probe_<model>.csv.
"""
import sys, glob, os
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
sys.path.insert(0, os.path.dirname(__file__))
from common import *


def load_emb(name, m):
    z = np.load(f'{EMB}/emb_{name}.npz', allow_pickle=True)  # our own files only
    key = dict(zip(zip(z['split'], z['ids']), range(len(z['ids']))))
    idx = np.array([key[(s, i)] for s, i in zip(m.split, m.Id)])
    return z['mean'][idx].astype(np.float32), z['std'][idx].astype(np.float32)


def novel_mask(cl, f):
    nov = np.zeros(len(cl), bool)
    for k in np.unique(f):
        va = f == k; trc = set(cl[~va])
        nov[va] = [c not in trc for c in cl[va]]
    return nov


def cv_probs(X, y, f, C=0.5):
    oof = np.zeros((len(y), 4))
    for k in np.unique(f):
        va = f == k
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))
        clf.fit(X[~va], y[~va]); oof[va] = clf.predict_proba(X[va])
    return oof


if __name__ == '__main__':
    m = load_meta(); tr = m.split.values == 'train'
    y = m.y.values[tr].astype(int); cl = m.cl.values[tr]
    f = folds(y, 0); nov = novel_mask(cl, f)
    print('novel rows', nov.sum(), 'of', len(y))
    for name in sys.argv[1:]:
        M, S = load_emb(name, m)
        L = M.shape[1]
        def one(l, use_std):
            X = np.concatenate([M[tr, l], S[tr, l]], 1) if use_std else M[tr, l]
            p = cv_probs(X, y, f, C=0.05 if X.shape[1] > 1500 else 0.1).argmax(1)
            return dict(model=name, layer=l, std=use_std, f1_novel=mf1(y[nov], p[nov]), f1_all=mf1(y, p))
        res = Parallel(n_jobs=4)(delayed(one)(l, s) for l in range(L) for s in ([False, True] if S.any() else [False]))
        R = pd.DataFrame(res); R.to_csv(f'{BH}/work/probe_{name}.csv', index=False)
        print(R.sort_values('f1_novel', ascending=False).head(6).to_string(index=False), flush=True)
