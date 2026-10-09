"""Train u-scorer on all OOF nodes, apply to test nodes, DP select, resolve, write CSV.
python make_sub.py --oof nodes/oof_f012.pkl nodes/oof_f34.pkl --test nodes/test_e2.pkl --tau 0.23 --floor 300 --out subs/f1.csv
"""
import sys, pickle, argparse
sys.path.insert(0, '/home/user/work/filament/code')
import numpy as np
from sel import *
from write_sub import resolve, to_rows, write

ap = argparse.ArgumentParser(); ap.add_argument('--oof', nargs='+'); ap.add_argument('--test'); ap.add_argument('--tau', type=float)
ap.add_argument('--floor', type=int, default=0); ap.add_argument('--resolve-floor', type=int, default=100); ap.add_argument('--one-piece', action='store_true')
ap.add_argument('--seeds', type=int, default=3); ap.add_argument('--out'); ap.add_argument('--feats', default='all')
a = ap.parse_args()
rows = sum([pickle.load(open(p, 'rb')) for p in a.oof], [])
T = [targets(r) for r in rows]; ms = [r['a2'] > 0 for r in rows]
X = np.concatenate([r['F'][m] for r, m in zip(rows, ms)]); y = np.concatenate([t[1][m] for t, m in zip(T, ms)])
models = [fit(X, y, 600, seed=s) for s in range(a.seeds)]
test = pickle.load(open(a.test, 'rb')); out = []; nsel = 0
for r in test:
    if len(r['rles']) == 0: continue
    sc = np.mean([m.predict(r['F']) for m in models], 0)
    sel = dp_select(r, sc, a.tau, valid=(r['a2'] >= max(a.floor, 1)))
    rl = resolve([r['rles'][i] for i in sel], sc[sel], floor=max(a.resolve_floor, a.floor), one_piece=a.one_piece)
    nsel += len(rl); out += to_rows(r['stem'], rl)
write(out, a.out); print(a.out, 'images', len(test), 'instances', nsel)
