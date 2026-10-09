"""Ensemble value on folds 1-4: r18 only vs r34 only vs mean(r18,r34), level-0.5 baseline filter, crossfit by fold."""
import sys, pickle
sys.path.insert(0, '/home/user/work/filament/code')
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sel import *

lv = lambda r: np.isclose(r['level'], 0.5) & (r['F'][:, 0] >= 30) & (r['a2'] > 0) if len(r['rles']) else np.zeros(0, bool)
for name, path in [('r18', 'oof_r18only_l5.pkl'), ('r34', 'oof_r34only_l5.pkl'), ('mean', 'oof_all.pkl')]:
    rows = [r for r in pickle.load(open('/home/user/work/filament/nodes/' + path, 'rb')) if FOLDS[r['stem']] != 0]
    fo = np.array([FOLDS[r['stem']] for r in rows]); T = [targets(r) for r in rows]
    pr = [np.zeros(len(r['rles'])) for r in rows]
    for f in range(1, 5):
        tr = [i for i in range(len(rows)) if fo[i] != f and lv(rows[i]).any()]
        m = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=30).fit(
            np.concatenate([rows[i]['F'][lv(rows[i]), :18] for i in tr]), np.concatenate([T[i][0][lv(rows[i])] for i in tr]))
        for i in range(len(rows)):
            if fo[i] == f and lv(rows[i]).any(): pr[i][lv(rows[i])] = m.predict(rows[i]['F'][lv(rows[i]), :18])
    for thr in [0.3, 0.35, 0.4]:
        print(name, thr, fmt(pq_sel(rows, [np.where(lv(r) & (p > thr))[0] for r, p in zip(rows, pr)])), flush=True)
