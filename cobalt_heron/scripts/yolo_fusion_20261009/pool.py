"""Joint candidate pool: U-Net level-0.5 CC nodes + YOLO instances, unified features, learned hit score, greedy disjoint selection.
python pool.py build-f0      -> nodes/pool_f0.pkl
python pool.py build-test E  -> nodes/pool_test_E.pkl (E: e2|e3, uses nodes/test_E.pkl + maps)
python pool.py eval
python pool.py sub E thr out.csv
"""
import sys, pickle, json
sys.path.insert(0, '/home/user/work/filament/code'); sys.path.insert(0, '/home/user/kraggle-3-in-one/cobalt_heron')
import numpy as np, cv2
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from pycocotools import mask as mu
from sklearn.ensemble import HistGradientBoostingRegressor
from sel import targets, FOLDS, pq_sel, fmt
from yolo_feats import load, node_yolo

W = Path('/home/user/work/filament')
lv = lambda r: np.isclose(r['level'], 0.5) & (r['F'][:, 0] >= 30) & (r['a2'] > 0) if len(r['rles']) else np.zeros(0, bool)
NF = 18 + 3 + 8


def cands(r, Y, pmap, gts):
    k = np.where(lv(r))[0]
    U = [r['rles'][i] for i in k]
    yl = [(a, c) for a, c in Y.get(r['stem'], []) if c >= 0.1]
    Yr = [a for a, _ in yl]; yc = np.array([c for _, c in yl])
    nu, ny = len(U), len(Yr)
    Fu = np.full((nu, NF), np.nan, np.float32); Fy = np.full((ny, NF), np.nan, np.float32)
    if nu:
        Fu[:, :18] = r['F'][k, :18]; Fu[:, 18:21] = node_yolo({'stem': r['stem'], 'rles': U}, Y)
    iuy = np.asarray(mu.iou(U, Yr, [0] * ny)).reshape(nu, ny) if nu and ny else np.zeros((nu, ny))
    p2 = cv2.resize(pmap, (2048, 2048), interpolation=cv2.INTER_LINEAR)
    allr = U + Yr
    for j, rl in enumerate(allr):
        m = mu.decode(rl).astype(bool); a = m.sum(); pv = p2[m] if a else np.zeros(1)
        isy = j >= nu
        conf = yc[j - nu] if isy else (Fu[j, 19] if nu else 0)
        if isy:
            bi = iuy[:, j - nu].max() if nu else 0
            yo = np.delete(np.arange(ny), j - nu); byy = np.asarray(mu.iou([rl], [Yr[t] for t in yo], [0] * len(yo))).max() if len(yo) else 0
        else:
            bi = iuy[j].max() if ny else 0; byy = 0
        row = [float(isy), conf, a, pv.mean(), pv.max(), (pv > 0.5).mean(), bi, byy]
        (Fy[j - nu] if isy else Fu[j])[21:] = row
    F = np.concatenate([Fu, Fy]) if nu + ny else np.zeros((0, NF), np.float32)
    ious = []
    if gts is not None:
        for g in gts:
            ious.append(np.asarray(mu.iou(allr, g, [0] * len(g)), np.float32).reshape(len(allr), len(g)) if len(g) and allr else np.zeros((len(allr), len(g)), np.float32))
    return dict(stem=r['stem'], rles=allr, F=F, ious=ious, isy=np.r_[np.zeros(nu), np.ones(ny)].astype(bool))


def _f0(r):
    from forest import gt2048
    G = gt2048(); Y = load(str(W / 'kout/ch-yolo-f0/out/f0_val_preds.json'))
    p = np.load(W / 'maps/oof/r34-f0' / f"{r['stem']}.npz")['p'][0].astype(np.float32) / 255
    return cands(r, Y, p, G[r['stem']])


def _te(args):
    r, dirs = args
    Y = load(str(W / 'kout/ch-yolo-full/out/full_test_preds.json'))
    p = np.mean([np.load(W / d / f"{r['stem']}.npz")['p'][0].astype(np.float32) / 255 for d in dirs], 0)
    return cands(r, Y, p, None)


def select(c, sc, thr, cov=0.2, floor=200):
    order = np.argsort(-sc); occ = np.zeros((2048, 2048), bool); out = []
    for i in order:
        if sc[i] <= thr: break
        m = mu.decode(c['rles'][i]).astype(bool); a = m.sum()
        if a == 0 or (m & occ).sum() / a >= cov: continue
        m &= ~occ
        if m.sum() < floor: continue
        occ |= m; out.append(mu.encode(np.asfortranarray(m.astype(np.uint8))))
    return out


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'build-f0':
        rows = [r for r in pickle.load(open(W / 'nodes/oof_all.pkl', 'rb')) if FOLDS[r['stem']] == 0]
        with ProcessPoolExecutor(int(sys.argv[2]) if len(sys.argv) > 2 else 3) as ex: out = list(ex.map(_f0, rows, chunksize=2))
        pickle.dump(out, open(W / 'nodes/pool_f0.pkl', 'wb')); print('saved', len(out), sum(len(c['rles']) for c in out))
    elif cmd == 'build-test':
        E = sys.argv[2]; dirs = {'e2': ['maps/test/r18s5', 'maps/test/r34f2'], 'e3': ['maps/test/r18s5', 'maps/test/r34f2', 'maps/test/r34-full']}[E]
        rows = pickle.load(open(W / f'nodes/test_{E}.pkl', 'rb'))
        with ProcessPoolExecutor(3) as ex: out = list(ex.map(_te, [(r, dirs) for r in rows], chunksize=2))
        pickle.dump(out, open(W / f'nodes/pool_test_{E}.pkl', 'wb')); print('saved', len(out))
    elif cmd in ('eval', 'sub'):
        C = pickle.load(open(W / 'nodes/pool_f0.pkl', 'rb'))
        # U-Net-only rows from folds 1-4 (YOLO-related feats NaN), as in the nan-mix filter
        oth = [r for r in pickle.load(open(W / 'nodes/oof_all.pkl', 'rb')) if FOLDS[r['stem']] != 0]
        Xo, yo = [], []
        for r in oth:
            m = lv(r)
            if not m.any(): continue
            F = np.full((m.sum(), NF), np.nan, np.float32); F[:, :18] = r['F'][m, :18]; F[:, 21] = 0; F[:, 23] = r['F'][m, 0] * 4
            Xo.append(F); yo.append(targets(r)[0][m])
        Xo, yo = np.concatenate(Xo), np.concatenate(yo)
        T = [targets(c)[0] for c in C]
        w0 = 3.0
        def fitm(idx):
            X = np.concatenate([Xo] + [C[i]['F'] for i in idx]); y = np.concatenate([yo] + [T[i] for i in idx])
            w = np.r_[np.ones(len(yo)), np.full(len(y) - len(yo), w0)]
            return HistGradientBoostingRegressor(max_iter=250, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=30).fit(X, y, sample_weight=w)
        if cmd == 'eval':
            months = sorted({c['stem'][:6] for c in C}); inner = np.array([months.index(c['stem'][:6]) % 4 for c in C])
            pr = [None] * len(C)
            for f in range(4):
                m = fitm([i for i in range(len(C)) if inner[i] != f and len(C[i]['rles'])])
                for i in range(len(C)):
                    if inner[i] == f: pr[i] = m.predict(C[i]['F']) if len(C[i]['rles']) else np.zeros(0)
            pickle.dump(pr, open(W / 'nodes/pool_f0_pred.pkl', 'wb'))
            from ch.metric import PQAccumulator
            from forest import gt2048
            G = gt2048()
            for thr in [0.3, 0.35, 0.4, 0.45]:
                for cov in [0.2]:
                    acc = PQAccumulator()
                    for c, p in zip(C, pr):
                        sel = select(c, p, thr, cov) if len(p) else []
                        for g in G[c['stem']]: acc.add(g, sel)
                    print('POOL', thr, cov, acc, flush=True)
        else:
            E, thr, out = sys.argv[2], float(sys.argv[3]), sys.argv[4]
            m = fitm([i for i in range(len(C)) if len(C[i]['rles'])])
            from write_sub import to_rows, write
            rows = []
            for c in pickle.load(open(W / f'nodes/pool_test_{E}.pkl', 'rb')):
                if len(c['rles']): rows += to_rows(c['stem'], select(c, m.predict(c['F']), thr))
            write(rows, out); print(out, len(rows))
