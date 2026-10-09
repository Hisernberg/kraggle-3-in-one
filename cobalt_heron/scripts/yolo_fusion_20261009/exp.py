import sys, pickle
sys.path.insert(0, '/home/user/work/filament/code')
import numpy as np
from sel import *

rows = pickle.load(open('/home/user/work/filament/nodes/oof_all.pkl', 'rb'))
fo = np.array([FOLDS[r['stem']] for r in rows])
T = [targets(r) for r in rows]


def cf(Xf, yf, mf, rounds=400):
    pr = [np.zeros(len(r['rles'])) for r in rows]
    for f in range(5):
        tr = [i for i in range(len(rows)) if fo[i] != f and len(rows[i]['rles'])]
        m = fit(np.concatenate([Xf(rows[i])[mf(rows[i])] for i in tr]), np.concatenate([yf(i)[mf(rows[i])] for i in tr]), rounds)
        for i in range(len(rows)):
            if fo[i] == f and len(rows[i]['rles']): pr[i] = m.predict(Xf(rows[i]))
    return pr


def ctx(r):
    """add parent feats + max child pmean/area"""
    F = r['F']; n = len(F)
    if n == 0: return np.zeros((0, F.shape[1] * 2 + 3), np.float32)
    P = np.where(r['parent'][:, None] >= 0, F[np.maximum(r['parent'], 0)], -1)
    cmx = np.zeros((n, 3))
    for i, p in enumerate(r['parent']):
        if p >= 0: cmx[p, 0] = max(cmx[p, 0], F[i, 1]); cmx[p, 1] = max(cmx[p, 1], F[i, 0]); cmx[p, 2] += F[i, 0]
    return np.concatenate([F, P, cmx], 1)


def sweep(name, scores, taus, levels=None):
    best = None
    for tau in taus:
        sels = []
        for r, s in zip(rows, scores):
            v = r['a2'] > 0
            if levels is not None: v &= np.isin(np.round(r['level'], 2), levels)
            sels.append(dp_select(r, s, tau, valid=v) if len(s) else [])
        d = pq_sel(rows, sels)
        if best is None or d['pq'] > best[1]['pq']: best = (tau, d)
        print(name, tau, fmt(d), flush=True)
    print('BEST', name, best[0], fmt(best[1]), flush=True)


mv = lambda r: r['a2'] > 0
which = sys.argv[1:]
if 'A' in which:
    l5 = lambda r: np.isclose(r['level'], 0.5) & (r['F'][:, 0] >= 30) & (r['a2'] > 0)
    h = cf(lambda r: r['F'], lambda i: T[i][0], l5)
    sweep('A_l5_lgbm_hit', h, [0.3, 0.35, 0.4, 0.45], levels=[0.5])
if 'C' in which:
    h = cf(lambda r: r['F'], lambda i: T[i][0], mv)
    pickle.dump(h, open('/home/user/work/filament/nodes/h_all.pkl', 'wb'))
    sweep('C_forest_hit', h, [0.3, 0.35, 0.4, 0.45, 0.5])
    for L in ([0.4, 0.5, 0.6], [0.3, 0.4, 0.5, 0.6, 0.7], [0.5, 0.6, 0.7, 0.8]):
        sweep(f'C_forest_hit_L{L}', h, [0.35, 0.4, 0.45], levels=L)
if 'D' in which:
    h = cf(ctx, lambda i: T[i][0], mv)
    pickle.dump(h, open('/home/user/work/filament/nodes/hctx_all.pkl', 'wb'))
    sweep('D_forest_hit_ctx', h, [0.3, 0.35, 0.4, 0.45, 0.5])
    u = cf(ctx, lambda i: T[i][1], mv)
    pickle.dump(u, open('/home/user/work/filament/nodes/uctx_all.pkl', 'wb'))
    sweep('D_forest_u_ctx', u, [0.19, 0.21, 0.23, 0.25, 0.27])
if 'B' in which:
    for L in [0.4, 0.6]:
        lm = lambda r, L=L: np.isclose(r['level'], L) & (r['F'][:, 0] >= 30) & (r['a2'] > 0)
        h = cf(lambda r: r['F'], lambda i: T[i][0], lm)
        sweep(f'B_l{L}_hit', h, [0.3, 0.35, 0.4, 0.45], levels=[L])
