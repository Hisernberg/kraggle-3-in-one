"""Fold-0 evaluation: scorer trained on folds1-4 (YOLO feats NaN) + fold-0 inner folds (YOLO feats), plus YOLO add rule (exact PQ)."""
import sys, pickle
sys.path.insert(0, '/home/user/work/filament/code'); sys.path.insert(0, '/home/user/kraggle-3-in-one/cobalt_heron')
import numpy as np
from sel import *
from yolo_feats import load, node_yolo
from sklearn.ensemble import HistGradientBoostingRegressor
from pycocotools import mask as mu
from ch.metric import PQAccumulator
from forest import gt2048
from write_sub import resolve

allr = pickle.load(open('/home/user/work/filament/nodes/oof_all.pkl', 'rb'))
Y = load('/home/user/work/filament/kout/ch-yolo-f0/out/f0_val_preds.json')
for r in allr: r['Y'] = node_yolo(r, Y) if FOLDS[r['stem']] == 0 else np.full((len(r['rles']), 3), np.nan, np.float32)
T = {r['stem']: targets(r) for r in allr}
f0 = [r for r in allr if FOLDS[r['stem']] == 0]; oth = [r for r in allr if FOLDS[r['stem']] != 0]
months = sorted({r['stem'][:6] for r in f0}); inner = np.array([months.index(r['stem'][:6]) % 4 for r in f0])
lv = lambda r: np.isclose(r['level'], 0.5) & (r['F'][:, 0] >= 30) & (r['a2'] > 0)
XY = lambda r: np.concatenate([r['F'][:, :18], r['Y']], 1)
XN = lambda r: r['F'][:, :18]


def cv(Xf, use_oth, w0=1.0):
    pr = [np.zeros(len(r['rles'])) for r in f0]
    for f in range(4):
        tr = [r for i, r in enumerate(f0) if inner[i] != f] + (oth if use_oth else [])
        tr = [r for r in tr if len(r['rles'])]
        X = np.concatenate([Xf(r)[lv(r)] for r in tr]); y = np.concatenate([T[r['stem']][0][lv(r)] for r in tr])
        w = np.concatenate([np.full(lv(r).sum(), w0 if FOLDS[r['stem']] == 0 else 1.0) for r in tr])
        m = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=30).fit(X, y, sample_weight=w)
        for i, r in enumerate(f0):
            if inner[i] == f and lv(r).any(): pr[i][lv(r)] = m.predict(Xf(r)[lv(r)])
    return pr


G = gt2048()


def exact(pr, thr, cadd=None, cov_max=0.2):
    acc = PQAccumulator()
    for r, p in zip(f0, pr):
        k = np.where(lv(r) & (p > thr))[0]
        rl = resolve([r['rles'][i] for i in k], p[k], floor=1) if len(k) else []
        if cadd is not None:
            occ = np.zeros((2048, 2048), bool)
            for x in rl: occ |= mu.decode(x).astype(bool)
            for yr, c in sorted(Y.get(r['stem'], []), key=lambda x: -x[1]):
                if c < cadd: continue
                m = mu.decode(yr).astype(bool); a = m.sum()
                if a == 0 or (m & occ).sum() / a >= cov_max: continue
                m &= ~occ
                if m.sum() < 200: continue
                occ |= m; rl.append(mu.encode(np.asfortranarray(m.astype(np.uint8))))
        for g in G[r['stem']]: acc.add(g, rl)
    return acc


pN_all = cv(XN, True); pY_all = cv(XY, True); pY_all3 = cv(XY, True, 3.0)
for thr in [0.3, 0.35, 0.4]:
    print('thr', thr, 'noY(all-data)', fmt(pq_sel(f0, [np.where(lv(r) & (p > thr))[0] for r, p in zip(f0, pN_all)])),
          '| Y nan-mix', fmt(pq_sel(f0, [np.where(lv(r) & (p > thr))[0] for r, p in zip(f0, pY_all)])),
          '| Y nan-mix w3', fmt(pq_sel(f0, [np.where(lv(r) & (p > thr))[0] for r, p in zip(f0, pY_all3)])), flush=True)
pickle.dump(dict(pN=pN_all, pY=pY_all, pY3=pY_all3), open('/home/user/work/filament/nodes/f0_yolo_preds.pkl', 'wb'))
for name, pr in [('noY', pN_all), ('Y', pY_all3)]:
    for cadd in [None, 0.5, 0.6, 0.7]:
        print('EXACT', name, 'thr0.35 add', cadd, exact(pr, 0.35, cadd), flush=True)
