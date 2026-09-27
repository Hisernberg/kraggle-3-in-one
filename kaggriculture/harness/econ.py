"""Instrumented game: per-player revenue/cost ledger by product (monkeypatches _commit_unit)."""
import sys, collections
from kaggle_environments.envs.kaggriculture import kaggriculture as K
from harness.fastsim import play

def ledger_game(a, b, seed=0):
    led = [collections.defaultdict(lambda: [0, 0.0]) for _ in range(2)]
    orig = K._commit_unit
    def patched(op, item, price, farm, private, market, shed_capacity=100):
        ok = orig(op, item, price, farm, private, market, shed_capacity)
        if ok:
            p = 0 if farm is patched.farms[0] else 1
            e = led[p][(op, item)]
            e[0] += 1; e[1] += price
        return ok
    K._commit_unit = patched
    orig_init = K._initialize
    def init(state, env):
        orig_init(state, env); patched.farms = state[0].observation.farms
    K._initialize = init
    try:
        r = play([a, b], seed=seed)
    finally:
        K._commit_unit = orig; K._initialize = orig_init
    return r, led

if __name__ == "__main__":
    r, led = ledger_game(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 0)
    print("money", r["money"], "shops", r["shops"])
    for p in range(2):
        print(f"--- player {p}")
        for (op, item), (n, v) in sorted(led[p].items(), key=lambda kv: -kv[1][1]):
            print(f"  {op:12s} {item:11s} n={n:6d} total={v:10.0f} avg={v/max(n,1):7.1f}")
