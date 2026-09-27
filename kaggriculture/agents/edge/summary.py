"""Summary table for edge candidates across the evaluation sets used in RESULTS.md."""
import sys, json, collections
sys.path.insert(0, '.')
from harness.tournament import outcome, _name
SETS = [('mirror 3000-3011', ['runs/edge_mirror.jsonl'], range(3000, 3012), 'cha22'),
        ('field 2000-2003', ['runs/edge_field2.jsonl', 'runs/rr_top.jsonl'], range(2000, 2004), None),
        ('field 2100-2103', ['runs/edge_fresh.jsonl'], range(2100, 2104), None),
        ('field 2200-2203', ['runs/edge_fresh3.jsonl'], range(2200, 2204), None),
        ('tetsutani 2000-2007', ['runs/edge_tet.jsonl', 'runs/edge_field2.jsonl', 'runs/rr_top.jsonl'], range(2000, 2008), 'tetsutani')]
heroes = sys.argv[1:]
def load(files):
    recs = {}
    for f in files:
        try:
            for l in open(f):
                if l.strip():
                    r = json.loads(l); recs.setdefault((r['a'], r['b'], r['seed']), r)
        except FileNotFoundError:
            pass
    return list(recs.values())
print(f"{'hero':26s}" + ''.join(f"{s[0]:>24s}" for s in SETS))
for h in heroes:
    row = []
    for label, files, seeds, only in SETS:
        W = G = 0.0; M = 0.0; seen = set()
        for r in load(files):
            if r['seed'] not in seeds: continue
            a, b = _name(r['a']), _name(r['b'])
            if h not in (a, b) or a == b: continue
            opp = b if a == h else a
            if only and only not in opp: continue
            if opp[:2] in ('e_', 'x_', 'y_', 'v_', 'w_', 'z_') or opp.startswith('cha22_edge'): continue
            key = (opp, r['seed'])
            if key in seen: continue  # count each (opponent, seed) once (games are seat-symmetric)
            seen.add(key)
            s = outcome(r); k = 0 if a == h else 1
            W += s if k == 0 else 1 - s; G += 1; M += r['money'][k] - r['money'][1 - k]
        if 'field' in label and 'cha22' in h:  # base vs itself: mirror tie
            W += 0.5 * len(seeds); G += len(seeds)
        row.append(f"{W:5.1f}/{int(G):<3d}({(M / G if G else 0):+6.0f})" if G else f"{'-':>18s}")
    print(f"{h[:26]:26s}" + ''.join(f"{x:>24s}" for x in row))
