"""Per-n min-L merge of candidate CSVs (float screen). Usage: merge.py out.csv base.csv cand1.csv ..."""
import sys, pandas as pd, numpy as np
sys.path.insert(0, __file__.rsplit('/', 1)[0])
from fastL import per_n
out, files = sys.argv[1], sys.argv[2:]
tabs = [per_n(f) for f in files]
best = {}
for k, t in enumerate(tabs):
    for n, r in t.iterrows():
        if r.cnt != n: continue
        if n not in best or r.L < best[n][0] - 1e-12: best[n] = (r.L, k)
src = pd.Series({n: v[1] for n, v in best.items()})
print('source counts', src.value_counts().to_dict())
parts = []
for k, f in enumerate(files):
    ns = set(src[src == k].index)
    if not ns: continue
    d = pd.read_csv(f, dtype=str); d = d[d.n.astype(int).isin(ns)]; parts.append(d)
m = pd.concat(parts); m['n'] = m.n.astype(int); m['square_id'] = m.square_id.astype(int)
m = m.sort_values(['n', 'square_id']).reset_index(drop=True); m['id'] = np.arange(len(m))
m[['id', 'n', 'square_id', 'x', 'y', 'angle']].to_csv(out, index=False)
print('merged fast score', round(sum(v[0] / np.sqrt(n) - 1 for n, v in best.items()), 7), 'n count', len(best))
for k, f in enumerate(files[1:], 1):
    for n in src[src == k].index: print(' n', n, 'from', f, 'gain', tabs[0].loc[n, 'L'] - best[n][0] if n in tabs[0].index else None)
