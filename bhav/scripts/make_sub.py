"""Build a submission from a JSON list of specs: python make_sub.py <out_name> '<json specs>' [weights]."""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from common import *
import blend

name, specs = sys.argv[1], json.loads(sys.argv[2])
w = json.loads(sys.argv[3]) if len(sys.argv) > 3 else [1] * len(specs)
m = load_meta(); O, T = [], []
for s in specs:
    oof, te = blend.run_spec(m, s); O.append(oof); T.append(te)
    print(s, blend.evaluate(m, oof).round(4), flush=True)
lo = sum(wi * np.log(o + 1e-9) for wi, o in zip(w, O)) / sum(w); oof = np.exp(lo)
lt = sum(wi * np.log(t + 1e-9) for wi, t in zip(w, T)) / sum(w); te = np.exp(lt)
print('BLEND novel/all', blend.evaluate(m, oof).round(4))
np.savez(f'{BH}/work/probs/{name}.npz', oof=oof, test=te)
t = blend.finalize(m, te); sub = blend.write_sub(t, f'{BH}/work/subs/{name}.csv')
print(t.src.value_counts().to_dict(), sub.emotion.value_counts().to_dict())
