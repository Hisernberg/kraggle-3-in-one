"""Exact host-metric OOF PQ after overlap resolution: exact_eval.py nodes.pkl[,nodes2.pkl] upkl tau floor [one_piece]"""
import sys, pickle
sys.path.insert(0, '/home/user/work/filament/code'); sys.path.insert(0, '/home/user/kraggle-3-in-one/cobalt_heron')
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from sel import dp_select, FOLDS
from write_sub import resolve
from ch.metric import PQAccumulator
from forest import gt2048

G = gt2048()
rows = sum([pickle.load(open(p, 'rb')) for p in sys.argv[1].split(',')], [])
up = sum([pickle.load(open(p, 'rb')) for p in sys.argv[2].split(',')], [])
tau, floor = float(sys.argv[3]), int(sys.argv[4]); op = len(sys.argv) > 5 and sys.argv[5] == '1'


def one(i):
    r, s = rows[i], up[i]; acc = PQAccumulator()
    sel = dp_select(r, s, tau, valid=(r['a2'] >= max(floor, 1))) if len(r['rles']) else []
    rl = resolve([r['rles'][j] for j in sel], s[sel], floor=max(100, floor), one_piece=op) if sel else []
    for g in G[r['stem']]: acc.add(g, rl)
    return FOLDS[r['stem']], acc


tot = PQAccumulator(); pf = {}
with ProcessPoolExecutor(4) as ex:
    for f, acc in ex.map(one, range(len(rows)), chunksize=4):
        tot.merge(acc); pf.setdefault(f, PQAccumulator()).merge(acc)
print('EXACT', sys.argv[3:], tot, 'perfold', ' '.join(f'{pf[f].pq:.3f}' for f in sorted(pf)))
