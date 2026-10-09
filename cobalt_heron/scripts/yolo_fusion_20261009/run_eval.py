import sys, pickle, json
sys.path.insert(0, '/home/user/work/filament/code')
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sel import *

rows = pickle.load(open(sys.argv[1], 'rb'))
rows = [r for r in rows]
print('images', len(rows), 'nodes', sum(len(r['rles']) for r in rows))
fo = np.array([FOLDS[r['stem']] for r in rows])

# --- baseline: s11 analogue (level 0.5 CC, area>=30, HistGBR on hit fraction, thr)
def base_preds(level=0.5):
    T = [targets(r) for r in rows]
    ms = [(np.isclose(r['level'], level)) & (r['F'][:, 0] >= 30) & (r['a2'] > 0) if len(r['rles']) else np.zeros(0, bool) for r in rows]
    pr = [np.zeros(len(r['rles'])) for r in rows]
    for f in sorted(set(fo)):
        tr = [i for i in range(len(rows)) if fo[i] != f]
        X = np.concatenate([rows[i]['F'][ms[i], :18] for i in tr]); y = np.concatenate([T[i][0][ms[i]] for i in tr])
        m = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=30).fit(X, y)
        for i in range(len(rows)):
            if fo[i] == f and ms[i].any(): pr[i][ms[i]] = m.predict(rows[i]['F'][ms[i], :18])
    return ms, pr

ms, bp = base_preds()
print('base all', fmt(pq_sel(rows, [np.where(m)[0] for m in ms])))
for th in [0.3, 0.35, 0.4]:
    print('base thr', th, fmt(pq_sel(rows, [np.where(m & (p > th))[0] for m, p in zip(ms, bp)])))

# --- forest + u-scorer + DP
up = crossfit(rows, 'u', mask_fn=lambda r: r['a2'] > 0)
pickle.dump(up, open(sys.argv[1] + '.u.pkl', 'wb'))
for floor in [0, 150, 300]:
    for tau in [0.16, 0.19, 0.21, 0.23, 0.25, 0.27, 0.30]:
        sels = [dp_select(r, s, tau, valid=(r['a2'] >= max(floor, 1))) for r, s in zip(rows, up)]
        d = pq_sel(rows, sels)
        print(f'forest floor{floor} tau{tau}', fmt(d), 'perfold', ' '.join(f"{pq_sel([rows[i] for i in range(len(rows)) if fo[i]==f], [sels[i] for i in range(len(rows)) if fo[i]==f])['pq']:.3f}" for f in range(5)), flush=True)
