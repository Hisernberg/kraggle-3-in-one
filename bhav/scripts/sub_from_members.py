"""Submission from cached members: python sub_from_members.py [--pool members.pkl] <name> <key1,key2,...> [extra probs npz ...]."""
import sys, os, pickle
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from common import *
import blend
from members import P

args = sys.argv[1:]
pool = P
if args[0] == '--pool':
    pool, args = args[1], args[2:]
name, keys, extra = args[0], args[1].split(','), args[2:]
R = pickle.load(open(pool, 'rb')); m = load_meta()
O = [R[k][0] for k in keys]; T = [R[k][1] for k in keys]
for f in extra:                       # extra members saved as npz (oof[3,N,4], test[Ntest,4])
    z = np.load(f); O.append(z['oof']); T.append(z['test'])
oof = np.exp(np.mean([np.log(o + 1e-9) for o in O], 0)); te = np.exp(np.mean([np.log(t + 1e-9) for t in T], 0))
print(name, 'CV novel/all', blend.evaluate(m, oof).round(4))
np.savez(f'{BH}/work/probs/{name}.npz', oof=oof, test=te)
t = blend.finalize(m, te); sub = blend.write_sub(t, f'{BH}/work/subs/{name}.csv')
print(t.src.value_counts().to_dict(), sub.emotion.value_counts().to_dict())
