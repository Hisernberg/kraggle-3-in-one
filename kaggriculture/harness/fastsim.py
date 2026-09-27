"""Fast, faithful Kaggriculture match runner.

Drives the official interpreter (kaggle_environments.envs.kaggriculture) directly,
skipping the framework's JSON-schema processing and per-step replay bookkeeping.
Agents are loaded the same way Kaggle does it (exec the file, take the last callable,
pass (obs, config) truncated to the callable's arg count), with a fresh namespace per game.

    python -m harness.fastsim agents/a/main.py agents/b/main.py --seed 7
"""
import argparse
import copy
import os
import sys
import time

from kaggle_environments.envs.kaggriculture import kaggriculture as K
from kaggle_environments.utils import Struct, structify

DEFAULT_CFG = {
    "episodeSteps": 720, "actTimeout": 1, "runTimeout": 1e9, "boardSize": 10, "startingMoney": 3000,
    "maxMarketOrdersPerTurn": 10, "turnsPerDay": 24, "shedCapacity": 100, "weedSpawnChance": 0.005,
    "townShopUnlockInterval": 3, "townShopSellInterval": 4, "townCenterSellInterval": 24,
    "farmHandCostMult": 1, "marketParams": {},
}


class _Env:
    def __init__(self, cfg, seed):
        self.configuration = structify(dict(cfg))
        self.info = {"seed": int(seed)}
        self.done = False


def load_agent(path):
    """Kaggle-style loader: exec the file, return the last callable in its namespace."""
    path = os.path.abspath(path)
    with open(path, encoding="utf-8") as f:
        raw = f.read()
    exec_dir = os.path.dirname(path)
    before = set(sys.modules)
    sys.path.insert(0, exec_dir)
    ns = {"__file__": path, "__name__": "__agent__"}
    try:
        exec(compile(raw, path, "exec"), ns)
    finally:
        sys.path.remove(exec_dir)
    fn = [v for v in ns.values() if callable(v)][-1]
    # Helper modules imported from the agent dir must not leak into the next agent.
    for m in set(sys.modules) - before:
        mod = sys.modules.get(m)
        f = getattr(mod, "__file__", None) or ""
        if f.startswith(exec_dir):
            del sys.modules[m]
    return fn


def _call(fn, obs, cfg):
    n = getattr(getattr(fn, "__code__", None), "co_argcount", 2)
    return fn(*([obs, cfg][:max(1, n)])) if n else fn()


def play(paths_or_fns, seed=0, cfg=None, record=False, verbose=False):
    """Play one game. Returns dict(rewards, status, times, [trace])."""
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    fns = [load_agent(p) if isinstance(p, str) else p for p in paths_or_fns]
    env = _Env(cfg, seed)
    state = [Struct(observation=Struct(player=i, step=0, remainingOverageTime=60, farms=[], private={},
                                       market={}, town={}, day=0, hour=0),
                    action=None, status="ACTIVE", reward=0, info={}) for i in range(2)]
    K.interpreter(state, env)
    cfg_s = structify(dict(cfg))
    status = ["ACTIVE", "ACTIVE"]
    times = [0.0, 0.0]
    maxt = [0.0, 0.0]
    errors = [None, None]
    trace = [] if record else None
    o0 = state[0].observation
    for t in range(cfg["episodeSteps"] - 1):
        o0.step = t
        actions = []
        for i in range(2):
            if status[i] != "ACTIVE":
                actions.append(None)
                continue
            ob = state[i].observation
            obs = structify({"player": i, "step": t, "day": o0.day, "hour": o0.hour, "farms": o0.farms,
                             "market": o0.market, "town": o0.town, "private": ob.private,
                             "remainingOverageTime": 60})
            t0 = time.perf_counter()
            try:
                a = _call(fns[i], obs, cfg_s)
            except Exception as e:  # Kaggle: ERROR -> reward None (a loss)
                import traceback
                errors[i] = f"t={t}: {e!r}\n" + traceback.format_exc()[-1500:]
                status[i] = "ERROR"
                a = None
            dt = time.perf_counter() - t0
            times[i] += dt
            maxt[i] = max(maxt[i], dt)
            if a is not None and not isinstance(a, dict):
                status[i] = "INVALID"
                errors[i] = f"t={t}: non-dict action {a!r:.200}"
                a = None
            actions.append(a)
        for i in range(2):
            state[i].action = structify(copy.deepcopy(actions[i])) if actions[i] is not None else \
                {"farmer": ["PASS"], "hands": [], "market": []}
        if record:
            trace.append({"t": t, "actions": copy.deepcopy(actions),
                          "money": [f["money"] for f in o0.farms],
                          "prices": dict(o0.market["prices"])})
        K.interpreter(state, env)
        if state[0].status == "DONE":
            break
    money = [float(f["money"]) for f in o0.farms]
    rewards = [money[i] if status[i] == "ACTIVE" else None for i in range(2)]
    out = {"rewards": rewards, "money": money, "status": status, "time": times, "max_turn_time": maxt,
           "errors": errors, "shops": list(o0.town["unlocked_shops"]), "seed": seed}
    if record:
        out["trace"] = trace
        out["final_state"] = {"farms": copy.deepcopy(o0.farms), "market": copy.deepcopy(o0.market),
                              "private": [copy.deepcopy(state[i].observation.private) for i in range(2)]}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time()
    r = play([args.a, args.b], seed=args.seed)
    r.pop("trace", None)
    print(r, f"{time.time() - t0:.1f}s")
