"""Per-day action-type histogram for one agent (seat 0) to spot wasted unit turns."""
import sys, collections
sys.path.insert(0, '.')
from harness.fastsim import load_agent, play

def run(a, b, seed, days):
    fn = load_agent(a)
    per = collections.defaultdict(collections.Counter)
    samples = collections.defaultdict(list)
    def spy(obs, cfg=None):
        act = fn(obs, cfg)
        d = obs['day']
        for u in [act.get('farmer')] + list(act.get('hands') or []):
            per[d][u[0] if u else 'NONE'] += 1
        if d in days and obs['hour'] in (6, 12):
            samples[d].append((obs['hour'], act, {k: v for k, v in obs['private']['shed'].items() if v}, obs['private']['inventories'], obs['farms'][obs['player']]['farmer'], obs['farms'][obs['player']]['hands']))
        return act
    r = play([spy, b], seed=seed)
    for d in sorted(per):
        print(d, dict(per[d].most_common()))
    for d in days:
        for s in samples[d]:
            print('SAMPLE', d, s)
    print(r['money'])

if __name__ == '__main__':
    run(sys.argv[1], sys.argv[2], int(sys.argv[3]), [int(x) for x in sys.argv[4].split(',')] if len(sys.argv) > 4 else [])
