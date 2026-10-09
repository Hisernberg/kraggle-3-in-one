import sys, pandas as pd, numpy as np
sys.path.insert(0, '/home/user/kraggle-3-in-one/cobalt_heron')
from ch.metric import PQAccumulator
def L(p):
    d = pd.read_csv(p, dtype=str); d['s'] = d.filament_id.str.rsplit('_', n=1).str[0]
    return {s: [{'size': [2048, 2048], 'counts': c.encode()} for c in g.segmentation_rle] for s, g in d.groupby('s')}
A = L(sys.argv[1])
for b in sys.argv[2:]:
    B = L(b); acc = PQAccumulator()
    for s in set(A) | set(B): acc.add(A.get(s, []), B.get(s, []))
    print(sys.argv[1], 'vs', b, acc)
