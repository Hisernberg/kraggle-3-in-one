"""Second-level stacking: multinomial LR on the members' OOF log-probabilities.

For each fold seed s the stacker is evaluated by an inner 5-fold CV (fold seed s+10) over the level-1 OOF of seed s,
so no row is scored by a stacker that saw it. Test: stacker fit on all level-1 OOF rows of every seed, applied to
the members' full-train test probabilities.
"""
import sys, os, pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, os.path.dirname(__file__))
from common import *
import blend
from members import P


def stack(R, keys, C=0.1):
    m = load_meta(); y = m.y.values[m.split.values == 'train'].astype(int)
    Z = lambda part, si=None: np.concatenate([np.log((R[k][part] if si is None else R[k][part][si]) + 1e-9) for k in keys], 1)
    oof2 = np.zeros((len(blend.SEEDS), len(y), 4))
    for si, s in enumerate(blend.SEEDS):
        X = Z(0, si); f = folds(y, s + 10)
        for k in range(5):
            va = f == k
            oof2[si, va] = LogisticRegression(C=C, max_iter=2000).fit(X[~va], y[~va]).predict_proba(X[va])
    Xall = np.vstack([Z(0, si) for si in range(len(blend.SEEDS))]); yall = np.tile(y, len(blend.SEEDS))
    test = LogisticRegression(C=C, max_iter=2000).fit(Xall, yall).predict_proba(Z(1))
    return oof2, test


if __name__ == '__main__':
    R = pickle.load(open(P, 'rb')); m = load_meta()
    for f in sys.argv[2:]:
        z = np.load(f); R[os.path.basename(f)[:-4]] = (z['oof'], z['test'])
    keys = sys.argv[1].split(',') if sys.argv[1] != 'all' else list(R)
    for C in [0.01, 0.03, 0.1, 0.3]:
        oof2, test = stack(R, keys, C); print('stack C', C, keys if len(keys) < 8 else len(keys), blend.evaluate(m, oof2).round(4), flush=True)
