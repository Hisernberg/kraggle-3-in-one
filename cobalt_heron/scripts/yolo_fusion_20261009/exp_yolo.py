"""Fold-0 YOLO evidence test: scorer with vs without YOLO feats (inner 4-fold CV by month within fold 0), plus s14-style add rule."""
import sys, pickle
sys.path.insert(0, '/home/user/work/filament/code')
import numpy as np
from sel import *
from yolo_feats import load, node_yolo
from sklearn.ensemble import HistGradientBoostingRegressor
from pycocotools import mask as mu

rows = [r for r in pickle.load(open('/home/user/work/filament/nodes/oof_all.pkl', 'rb')) if FOLDS[r['stem']] == 0]
Y = load('/home/user/work/filament/kout/ch-yolo-f0/out/f0_val_preds.json')
for r in rows: r['Y'] = node_yolo(r, Y)
T = [targets(r) for r in rows]
months = sorted({r['stem'][:6] for r in rows}); inner = np.array([months.index(r['stem'][:6]) % 4 for r in rows])
lv = lambda r: np.isclose(r['level'], 0.5) & (r['F'][:, 0] >= 30) & (r['a2'] > 0)


def cv(Xf, msk):
    pr = [np.zeros(len(r['rles'])) for r in rows]
    for f in range(4):
        tr = [i for i in range(len(rows)) if inner[i] != f and len(rows[i]['rles'])]
        X = np.concatenate([Xf(rows[i])[msk(rows[i])] for i in tr]); y = np.concatenate([T[i][0][msk(rows[i])] for i in tr])
        m = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=30).fit(X, y)
        for i in range(len(rows)):
            if inner[i] == f and msk(rows[i]).any(): pr[i][msk(rows[i])] = m.predict(Xf(rows[i])[msk(rows[i])])
    return pr


def ev(pr, thr, msk, add=None):
    sels = [np.where(msk(r) & (p > thr))[0] for r, p in zip(rows, pr)]
    return pq_sel(rows, sels)


p0 = cv(lambda r: r['F'][:, :18], lv)
p1 = cv(lambda r: np.concatenate([r['F'][:, :18], r['Y']], 1), lv)
for thr in [0.3, 0.35, 0.4, 0.45]:
    print('thr', thr, 'noY', fmt(ev(p0, thr, lv)), '| withY', fmt(ev(p1, thr, lv)), flush=True)

# YOLO alone (disjoint greedy) for reference via exact metric
from yolo_feats import load as _l
from ch.metric import PQAccumulator
from forest import gt2048
G = gt2048()
for c in [0.3, 0.4, 0.5]:
    acc = PQAccumulator()
    for r in rows:
        items = sorted([x for x in Y.get(r['stem'], []) if x[1] >= c], key=lambda x: -x[1])
        occ = None; pr = []
        for rl, cf in items:
            m = mu.decode(rl).astype(bool)
            if occ is None: occ = np.zeros_like(m)
            m &= ~occ
            if m.sum() < 200: continue
            occ |= m; pr.append(mu.encode(np.asfortranarray(m.astype(np.uint8))))
        for g in G[r['stem']]: acc.add(g, pr)
    print('yolo alone conf', c, acc, flush=True)
