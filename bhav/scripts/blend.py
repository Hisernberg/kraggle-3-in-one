"""OOF + test probabilities for feature sets, blending, duplicate override, submission writer."""
import sys, os, json
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
sys.path.insert(0, os.path.dirname(__file__))
from common import *
from probe import load_emb, novel_mask

SEEDS = [0, 1, 2]


def features(m, spec):
    """spec: {'model': name, 'layers': [..], 'std': bool}; layers are averaged (mean of pooled vectors)."""
    M, S = load_emb(spec['model'], m)
    L = spec['layers']
    X = M[:, L].mean(1)
    if spec.get('std'):
        X = np.concatenate([X, S[:, L].mean(1)], 1)
    if spec.get('spknorm'):
        X = speaker_norm(X, spec['spknorm'], spec.get('spkmin', 5), spec.get('spkscale', False))
    return X


def speaker_norm(X, th, min_size, scale):
    """Subtract the mean (optionally divide by the std) of each ECAPA speaker cluster, train and test clips
    together (transductive, no labels). Clusters smaller than min_size are left as they are."""
    c = np.load(f'{EMB}/spk_cl_{th}.npy'); X = X.astype(np.float64).copy()
    for k in np.unique(c):
        i = c == k
        if i.sum() >= min_size:
            X[i] -= X[i].mean(0)
            if scale:
                X[i] /= X[i].std(0) + 1e-3
    return X


def make_clf(kind, C):
    if kind == 'lr':
        return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000))
    if kind == 'svm':
        return make_pipeline(StandardScaler(), SVC(C=C, kernel='rbf', probability=True, random_state=0))
    raise ValueError(kind)


def run_spec(m, spec):
    """Return oof[seed,N,4] on train and test[Ntest,4] fitted on all train."""
    X = features(m, spec); tr = (m.split == 'train').values
    y = m.y.values[tr].astype(int); Xtr, Xte = X[tr], X[~tr]
    kind, C = spec.get('clf', 'lr'), spec.get('C', 0.1)
    def fold_job(seed, k):
        f = folds(y, seed); va = f == k
        clf = make_clf(kind, C).fit(Xtr[~va], y[~va]); return seed, va, clf.predict_proba(Xtr[va])
    res = Parallel(n_jobs=4)(delayed(fold_job)(s, k) for s in SEEDS for k in range(5))
    oof = np.zeros((len(SEEDS), len(y), 4))
    for s, va, p in res:
        oof[SEEDS.index(s), va] = p
    test = make_clf(kind, C).fit(Xtr, y).predict_proba(Xte)
    return oof, test


def evaluate(m, oof, bias=None):
    """Mean over seeds of: model-only F1 on novel rows, and full-pipeline F1 (override + model)."""
    tr = (m.split == 'train').values; y = m.y.values[tr].astype(int); cl = m.cl.values[tr]
    b = np.zeros(4) if bias is None else bias
    out = []
    for si, s in enumerate(SEEDS):
        f = folds(y, s); nov = novel_mask(cl, f); p = np.log(oof[si] + 1e-9) + b
        pred = p.argmax(1).copy()
        for k in range(5):
            va = np.where(f == k)[0]
            for i in va:
                same = (cl == cl[i]) & (f != k)
                if same.any():
                    pred[i] = np.bincount(y[same], minlength=4).argmax()
        out.append((mf1(y[nov], p[nov].argmax(1)), mf1(y, pred)))
    return np.array(out).mean(0)


def finalize(m, test_prob, bias=None):
    """Test labels: train-duplicate label when one exists, else cluster-averaged model probabilities."""
    tr = (m.split == 'train').values; te = m[~tr].reset_index(drop=True)
    ytr = m.y.values[tr].astype(int); cltr = m.cl.values[tr]
    lp = np.log(test_prob + 1e-9) + (0 if bias is None else bias)
    lp = pd.DataFrame(lp).groupby(te.cl.values).transform('mean').values   # test-test duplicates agree
    pred = lp.argmax(1); src = np.array(['model'] * len(te), dtype=object)
    for i, c in enumerate(te.cl.values):
        same = cltr == c
        if same.any():
            pred[i] = np.bincount(ytr[same], minlength=4).argmax(); src[i] = 'dup'
    return te.assign(emotion=[CLASSES[p] for p in pred], src=src)


def write_sub(te, path):
    sub = pd.read_csv(f'{DATA}/sample_submission.csv')[['Id']]
    sub = sub.merge(te[['Id', 'emotion']], on='Id', how='left')
    assert sub.emotion.notna().all() and len(sub) == 209 and set(sub.emotion) <= set(CLASSES)
    sub.to_csv(path, index=False)
    return sub
