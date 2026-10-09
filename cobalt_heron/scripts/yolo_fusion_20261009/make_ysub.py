"""Level-0.5 CC + HistGBR hit filter with YOLO evidence feats (fold-0 rows have YOLO-f0 feats, others NaN, weight w0),
applied to test nodes with YOLO-full test feats; optional YOLO add rule; resolve; CSV."""
import sys, pickle, argparse
sys.path.insert(0, '/home/user/work/filament/code')
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from pycocotools import mask as mu
from sel import targets, FOLDS
from yolo_feats import load, node_yolo
from write_sub import resolve, to_rows, write

ap = argparse.ArgumentParser(); ap.add_argument('--test'); ap.add_argument('--thr', type=float, default=0.35); ap.add_argument('--w0', type=float, default=3.0)
ap.add_argument('--cadd', type=float, default=None); ap.add_argument('--floor', type=int, default=0); ap.add_argument('--one-piece', action='store_true')
ap.add_argument('--noyolo', action='store_true'); ap.add_argument('--seeds', type=int, default=1); ap.add_argument('--out')
a = ap.parse_args()
rows = pickle.load(open('/home/user/work/filament/nodes/oof_all.pkl', 'rb'))
Yf0 = load('/home/user/work/filament/kout/ch-yolo-f0/out/f0_val_preds.json')
Yte = load('/home/user/work/filament/kout/ch-yolo-full/out/full_test_preds.json')
lv = lambda r: np.isclose(r['level'], 0.5) & (r['F'][:, 0] >= 30) & (r['a2'] > 0) if len(r['rles']) else np.zeros(0, bool)
def feats(r, Y):
    if a.noyolo: return r['F'][:, :18]
    yf = node_yolo(r, Y) if Y is not None else np.full((len(r['rles']), 3), np.nan, np.float32)
    return np.concatenate([r['F'][:, :18], yf], 1)
X, y, w = [], [], []
for r in rows:
    m = lv(r)
    if not m.any(): continue
    f0 = FOLDS[r['stem']] == 0
    X.append(feats(r, Yf0 if f0 else None)[m]); y.append(targets(r)[0][m]); w.append(np.full(m.sum(), a.w0 if f0 else 1.0))
X, y, w = np.concatenate(X), np.concatenate(y), np.concatenate(w)
models = [HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=30, random_state=s).fit(X, y, sample_weight=w) for s in range(a.seeds)]
test = pickle.load(open(a.test, 'rb')); out = []
for r in test:
    k = np.where(lv(r))[0]; rl = []
    if len(k):
        sc = np.mean([m.predict(feats(r, Yte)[k]) for m in models], 0)
        k2 = k[sc > a.thr]; sc = sc[sc > a.thr]
        rl = resolve([r['rles'][i] for i in k2], sc, floor=max(1, a.floor), one_piece=a.one_piece)
    if a.cadd is not None:
        occ = np.zeros((2048, 2048), bool)
        for x in rl: occ |= mu.decode(x).astype(bool)
        for yr, c in sorted(Yte.get(r['stem'], []), key=lambda x: -x[1]):
            if c < a.cadd: continue
            m = mu.decode(yr).astype(bool); ar = m.sum()
            if ar == 0 or (m & occ).sum() / ar >= 0.2: continue
            m &= ~occ
            if m.sum() < max(200, a.floor): continue
            occ |= m; rl.append(mu.encode(np.asfortranarray(m.astype(np.uint8))))
    out += to_rows(r['stem'], rl)
write(out, a.out); print(a.out, len(out))
