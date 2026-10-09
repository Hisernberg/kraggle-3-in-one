"""s11-recipe analogue: level-0.5 CC (merge4, area>=30@1024), HistGBR hit filter trained on all OOF, thr, 2048 x1.2 upsample.
python make_base.py --oof nodes/oof_all.pkl --test nodes/test_e2.pkl --thr 0.35 --out subs/x.csv [--floor 0]"""
import sys, pickle, argparse
sys.path.insert(0, '/home/user/work/filament/code')
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sel import targets
from write_sub import resolve, to_rows, write

ap = argparse.ArgumentParser(); ap.add_argument('--oof', nargs='+'); ap.add_argument('--test'); ap.add_argument('--thr', type=float, default=0.35)
ap.add_argument('--floor', type=int, default=0); ap.add_argument('--level', type=float, default=0.5); ap.add_argument('--out'); ap.add_argument('--one-piece', action='store_true')
a = ap.parse_args()
rows = sum([pickle.load(open(p, 'rb')) for p in a.oof], [])
msk = lambda r: np.isclose(r['level'], a.level) & (r['F'][:, 0] >= 30) & (r['a2'] > 0) if len(r['rles']) else np.zeros(0, bool)
X = np.concatenate([r['F'][msk(r), :18] for r in rows]); y = np.concatenate([targets(r)[0][msk(r)] for r in rows])
m = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=30).fit(X, y)
test = pickle.load(open(a.test, 'rb')); out = []
for r in test:
    if not len(r['rles']): continue
    k = np.where(msk(r))[0]
    if not len(k): continue
    sc = m.predict(r['F'][k, :18]); k2 = k[sc > a.thr]; sc = sc[sc > a.thr]
    rl = resolve([r['rles'][i] for i in k2], sc, floor=max(1, a.floor), one_piece=a.one_piece)
    out += to_rows(r['stem'], rl)
write(out, a.out); print(a.out, len(out))
