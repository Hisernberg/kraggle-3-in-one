"""Scorer + selection on forest nodes; fast PQ from precomputed IoU matrices."""
import sys, json, pickle
sys.path.insert(0, '/home/user/kraggle-3-in-one/cobalt_heron')
import numpy as np
import lightgbm as lgb
from pycocotools import mask as mu

FOLDS = json.load(open('/home/user/work/filament/w/folds.json'))


def targets(r):
    """per node: h = mean_r hit, u = mean_r iou*hit"""
    n = len(r['rles']); R = max(len(r['ious']), 1)
    h = np.zeros(n); u = np.zeros(n)
    for M in r['ious']:
        if M.shape[1] == 0 or n == 0: continue
        b = M.max(1); hit = b > 0.5
        h += hit; u += np.where(hit, b, 0)
    return h / R, u / R


def pq_sel(rows, sels):
    s = tp = fp = fn = 0.0
    for r, S in zip(rows, sels):
        S = np.asarray(S, int)
        for M in r['ious']:
            ng = M.shape[1]
            if ng == 0: fp += len(S); continue
            if len(S) == 0: fn += ng; continue
            X = M[S]; hit = X > 0.5
            s += X[hit].sum(); tp += hit.sum(); fp += (hit.sum(1) == 0).sum(); fn += (hit.sum(0) == 0).sum()
    d = tp + 0.5 * fp + 0.5 * fn
    return dict(pq=s / d if d else 0, sq=s / tp if tp else 0, tp=int(tp), fp=int(fp), fn=int(fn))


def fmt(d): return f"PQ={d['pq']:.4f} SQ={d['sq']:.3f} TP={d['tp']} FP={d['fp']} FN={d['fn']}"


PARAMS = dict(objective='regression', learning_rate=0.03, num_leaves=31, min_data_in_leaf=40, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=1)


def fit(X, y, rounds=600, seed=0):
    p = dict(PARAMS, seed=seed)
    return lgb.train(p, lgb.Dataset(X, y), rounds)


def crossfit(rows, target='u', mask_fn=None, rounds=600, extra=None):
    """cross-fitted predictions by month-fold; returns list of arrays."""
    fo = np.array([FOLDS[r['stem']] for r in rows])
    T = [targets(r) for r in rows]
    Xs = [r['F'] if extra is None else extra(r) for r in rows]
    ys = [t[1] if target == 'u' else t[0] for t in T]
    ms = [np.ones(len(r['rles']), bool) if mask_fn is None else mask_fn(r) for r in rows]
    preds = [np.zeros(len(r['rles'])) for r in rows]
    for f in sorted(set(fo)):
        tr = [i for i in range(len(rows)) if fo[i] != f]
        X = np.concatenate([Xs[i][ms[i]] for i in tr]); y = np.concatenate([ys[i][ms[i]] for i in tr])
        m = fit(X, y, rounds)
        for i in range(len(rows)):
            if fo[i] == f and len(Xs[i]): preds[i] = m.predict(Xs[i])
    return preds


def children(r):
    ch = [[] for _ in range(len(r['rles']))]; roots = []
    for i, p in enumerate(r['parent']):
        (ch[p] if p >= 0 else roots).append(i)
    return ch, roots


def dp_select(r, score, tau, valid=None):
    ch, roots = children(r); n = len(score)
    valid = np.ones(n, bool) if valid is None else valid
    best = np.zeros(n); take = np.zeros(n, bool)
    order = np.argsort(-r['level'], kind='stable')  # deepest (highest level) first
    for v in order:
        cs = sum(best[c] for c in ch[v])
        own = score[v] - tau if valid[v] else -1e9
        if own > cs and own > 0: best[v] = own; take[v] = True
        else: best[v] = max(cs, 0.0)
    sel = []
    def walk(v):
        if take[v]: sel.append(v); return
        for c in ch[v]: walk(c)
    for rt in roots: walk(rt)
    return sel
