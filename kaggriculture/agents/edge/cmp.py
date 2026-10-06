"""Per-opponent comparison of heroes (incl. the base from a round-robin file).
    python agents/edge/cmp.py runs/a.jsonl runs/b.jsonl ... [--seeds 2000-2003]"""
import sys, json, collections
sys.path.insert(0, '.')
from harness.tournament import outcome, _name
args = [a for a in sys.argv[1:] if not a.startswith('--')]
seeds = None
for a in sys.argv[1:]:
    if a.startswith('--seeds='):
        lo, hi = a.split('=')[1].split('-'); seeds = set(range(int(lo), int(hi) + 1))
recs = []
for f in args:
    recs += [json.loads(l) for l in open(f) if l.strip()]
tab = collections.defaultdict(lambda: [0.0, 0, 0.0])
seen = set()
for r in recs:
    if seeds and r['seed'] not in seeds: continue
    key = (r['a'], r['b'], r['seed'])
    if key in seen: continue
    seen.add(key)
    s = outcome(r)
    if s is None: continue
    a, b = _name(r['a']), _name(r['b'])
    for who, opp, sc, k in ((a, b, s, 0), (b, a, 1 - s, 1)):
        t = tab[(who, opp)]; t[0] += sc; t[1] += 1; t[2] += r['money'][k] - r['money'][1 - k]
heroes = sorted({w for w, _ in tab if w[:2] in ('e_','x_','y_','z_','w_','v_') or 'cha22' in w})
opps = sorted({o for _, o in tab if o[:2] not in ('e_','x_','y_','z_','w_','v_')})
short = {o: (o.split("__")[0][:5] + ":" + o.split("__")[-1][:7]) if "__" in o else o[:13] for o in opps}
print(f"{'hero':24s} {'total':>13s} " + ' '.join(f"{short[o]:>13s}" for o in opps))
for h in heroes:
    row = []; W = G = 0; M = 0.0
    for o in opps:
        t = tab.get((h, o))
        if o == h or not t: row.append(f"{'-':>13s}"); continue
        W += t[0]; G += t[1]; M += t[2]
        row.append(f"{t[0]:4.1f}/{t[1]:<2d}{t[2]/t[1]:+6.0f}")
    print(f"{h[:24]:24s} {W:5.1f}/{G:<3d}{(M/G if G else 0):+5.0f} " + ' '.join(row))
