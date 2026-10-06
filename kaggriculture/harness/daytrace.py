"""Per-day diagnostic of one player's farm: animals, plants by crop, weeds, empties, hands, money, shed."""
import sys, collections
sys.path.insert(0, '.')
from harness.fastsim import load_agent, play

def run(a, b, seed, who=0):
    fn = load_agent(a)
    rows = []
    def spy(obs, cfg=None):
        act = fn(obs, cfg)
        if obs['hour'] in (0, 23):
            f = obs['farms'][obs['player']]
            c = collections.Counter()
            for row in f['tiles']:
                for t in row:
                    if t is None: c['empty'] += 1
                    elif t == 'LOCKED': pass
                    elif t['kind'] == 'PLANT': c[t['crop'][:5]] += 1
                    elif t['kind'] == 'WEED': c['weed'] += 1
                    elif t.get('animal'): c[t['animal']] += 1; c['unfed'] += (not t['fed_today']); c['uncared'] += (not t['cared_today'])
                    else: c['struct'] += 1
            unw = sum(1 for row in f['tiles'] for t in row if isinstance(t, dict) and t.get('kind') == 'PLANT' and not t['watered_today'])
            sh = {k: v for k, v in obs['private']['shed'].items() if v}
            rows.append(f"d{obs['day']:2d}h{obs['hour']:2d} ${f['money']:8.0f} hands={len(f['hands'])} unwatered={unw} {dict(c)} shed={sh} seeds={ {k:v for k,v in obs['private']['seeds'].items() if v} }")
        return act
    seats = [spy, b] if who == 0 else [b, spy]
    r = play(seats, seed=seed)
    print("\n".join(rows)); print(r['money'], r['errors'])

if __name__ == '__main__':
    run(sys.argv[1], sys.argv[2], int(sys.argv[3]))
