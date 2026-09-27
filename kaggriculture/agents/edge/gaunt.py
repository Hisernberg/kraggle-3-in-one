"""Single-seat gauntlet (games are seat-symmetric in practice): hero as seat 0 vs each opponent.
    python agents/edge/gaunt.py --hero H1 H2 --opps-file F --seeds N --seed0 S --workers 2 --out runs/x.jsonl"""
import argparse, json, os, sys, time, multiprocessing as mp
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from harness.tournament import _job
ap = argparse.ArgumentParser()
ap.add_argument('--hero', nargs='+'); ap.add_argument('--opps-file'); ap.add_argument('--opps', nargs='*', default=[])
ap.add_argument('--seeds', type=int, default=4); ap.add_argument('--seed0', type=int, default=2000)
ap.add_argument('--workers', type=int, default=2); ap.add_argument('--out', required=True)
a = ap.parse_args()
opps = list(a.opps) + ([l.strip() for l in open(a.opps_file) if l.strip() and not l.startswith('#')] if a.opps_file else [])
jobs = [(h, o, s) for h in a.hero for s in range(a.seed0, a.seed0 + a.seeds) for o in opps if o != h]
have = set()
if os.path.exists(a.out):
    for l in open(a.out):
        if l.strip():
            r = json.loads(l); have.add((r['a'], r['b'], r['seed'])); have.add((r['b'], r['a'], r['seed']))
todo = [j for j in jobs if j not in have]
print(len(todo), 'games', flush=True)
t0 = time.time()
with mp.get_context('fork').Pool(a.workers, maxtasksperchild=1) as pool, open(a.out, 'a') as f:
    for k, rec in enumerate(pool.imap_unordered(_job, todo), 1):
        f.write(json.dumps(rec) + '\n'); f.flush()
        if k % 20 == 0: print(k, f'{time.time()-t0:.0f}s', flush=True)
