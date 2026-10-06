"""Kaggriculture replay parser.

Turns a Kaggle replay JSON (env.toJSON() format, as downloaded by
`kaggle competitions replay <id>`) into per-player timelines and summary
features.

Replay layout: replay["steps"][t][p] = {action, observation, reward, status}.
The action stored at steps[t] is the one the agent returned for observation
t-1, i.e. it was processed by the interpreter at game step s = t-1
(day = s // 24, hour = s % 24) and produced the state stored at steps[t].

Market orders are re-simulated exactly with the engine's own pricing and
commit functions (unit actions -> lockstep market -> compare with recorded
money), so every SELL / BUY fill gets its exact per-unit price.

Usage:
    python replay_parse.py REPLAY.json [--json out.json] [--print]
    python replay_parse.py DIR --table          # one line per replay
"""
import copy
import glob
import json
import os
import sys
from collections import Counter, defaultdict

ENGINE_DIR = "/usr/local/lib/python3.11/dist-packages/kaggle_environments/envs/kaggriculture"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from kaggle_environments.envs.kaggriculture import kaggriculture as E
except Exception:  # pragma: no cover - fallback to local copy
    import engine_kaggriculture as E

TPD = 24
BOARD = 10
CROPS = list(E.CROPS)
ANIMALS = list(E.ANIMALS)


# ----------------------------------------------------------------- helpers
def _obs(step, p):
    return step[p]["observation"]


def _shared(step):
    """Shared fields (farms/market/town) live on player 0's observation."""
    return step[0]["observation"]


def tile_counts(tiles):
    c = Counter()
    for row in tiles:
        for t in row:
            if t is None:
                c["EMPTY"] += 1
            elif t == "LOCKED":
                c["LOCKED"] += 1
            elif isinstance(t, dict):
                k = t.get("kind")
                if k == "PLANT":
                    c[t["crop"]] += 1
                elif k == "WEED":
                    c["WEED"] += 1
                elif "animal" in t:
                    c[t["animal"]] += 1
                else:
                    c[k + "_EMPTY"] += 1
    return dict(c)


def layout_str(tiles):
    return "\n".join(" ".join(E._render_tile(t) for t in row) for row in tiles)


def act_str(a):
    if not isinstance(a, list) or not a:
        return "PASS"
    return " ".join(str(x) for x in a)


# --------------------------------------------------- exact market re-sim
def simulate_market(prev_step, step, cfg):
    """Re-simulate one interpreter call's unit actions + market phase.

    Returns (fills, money_after) where fills is a list per player of dicts
    {op, item, req, filled, prices[list], idx}.
    """
    farms = copy.deepcopy(_shared(prev_step)["farms"])
    market = copy.deepcopy(_shared(prev_step)["market"])
    privs = []
    for p in range(2):
        pv = _obs(prev_step, p).get("private")
        privs.append(copy.deepcopy(pv) if pv else None)
    if privs[0] is None or privs[1] is None:
        return None, None
    s = _shared(prev_step).get("step", 0) or 0
    day = s // TPD
    shed_cap = int(cfg.get("shedCapacity", 100))
    # unit actions (affect shed via DROP/PLACE/PICKUP before the market)
    for p in range(2):
        action = step[p].get("action") or {}
        if not isinstance(action, dict):
            action = {}
        fa = action.get("farmer", ["PASS"])
        ha = action.get("hands", []) or []
        units = [fa, *ha]
        demand = Counter(a[1] for a in units if isinstance(a, list) and len(a) >= 2 and a[0] == "PLANT")
        blocked = {c for c, n in demand.items() if n > privs[p]["seeds"].get(c, 0)}
        for idx, a in enumerate(units):
            if isinstance(a, list) and len(a) >= 2 and a[0] == "PLANT" and a[1] in blocked:
                a = ["PASS"]
            E._apply_unit_action(farms[p], privs[p], idx, a, BOARD, day, TPD, shed_cap)
    # market phase (copy of E._process_market, instrumented)
    queues = []
    for p in range(2):
        action = step[p].get("action") or {}
        m = action.get("market", []) if isinstance(action, dict) else []
        queues.append(list(m)[: int(cfg.get("maxMarketOrdersPerTurn", 10))] if isinstance(m, list) else [])
    fills = [[], []]
    max_len = max(len(q) for q in queues) if queues else 0
    for i in range(max_len):
        ostates = []
        for p, q in enumerate(queues):
            o = E._parse_order(q[i]) if i < len(q) else None
            if o is not None:
                o["rec"] = {"op": o["type"], "item": o.get("item"), "req": o.get("remaining", 1),
                            "filled": 0, "prices": [], "idx": i}
                fills[p].append(o["rec"])
            ostates.append(o)
        for p, o in enumerate(ostates):
            if o is None:
                continue
            if o["type"] == "HIRE":
                before = farms[p]["money"]
                E._do_hire(farms[p], privs[p], BOARD)
                if farms[p]["money"] < before:
                    o["rec"]["filled"] = 1
                    o["rec"]["prices"].append(before - farms[p]["money"])
                ostates[p] = None
            elif o["type"] == "BUY_LAND":
                before = farms[p]["money"]
                E._do_buy_land(farms[p], BOARD)
                if farms[p]["money"] < before:
                    o["rec"]["filled"] = 1
                    o["rec"]["prices"].append(before - farms[p]["money"])
                ostates[p] = None
        guard = 0
        while guard < 100000:
            guard += 1
            quoted = [None, None]
            for p, o in enumerate(ostates):
                if o is None or o["remaining"] <= 0:
                    continue
                op, item = o["type"], o["item"]
                if op == "SELL" and item in E.PRODUCTS:
                    quoted[p] = (op, item, E.market_price(item, market["inventory"][item], market.get("params")), o)
                elif op == "BUY_PRODUCT" and item in ("WHEAT", "FERTILIZER"):
                    quoted[p] = (op, item, E.market_price(item, market["inventory"][item] - 1, market.get("params")), o)
                elif op == "BUY_SEED" and item in E.CROPS:
                    quoted[p] = (op, item, E.CROPS[item]["seed"], o)
                elif op == "BUY_ANIMAL" and item in E.ANIMALS:
                    quoted[p] = (op, item, E.ANIMALS[item]["cost"], o)
                else:
                    ostates[p] = None
            if all(q is None for q in quoted):
                break
            any_ok = False
            for p, q in enumerate(quoted):
                if q is None:
                    continue
                op, item, price, o = q
                if E._commit_unit(op, item, price, farms[p], privs[p], market, shed_cap):
                    o["remaining"] -= 1
                    o["rec"]["filled"] += 1
                    o["rec"]["prices"].append(price)
                    any_ok = True
                else:
                    ostates[p] = None
            if not any_ok:
                break
        E._refresh_prices(market)
    return fills, [f["money"] for f in farms]


# ------------------------------------------------------------------ parse
def parse_replay(path):
    with open(path) as f:
        rep = json.load(f)
    steps = rep["steps"]
    cfg = rep.get("configuration", {}) or {}
    info = rep.get("info", {}) or {}
    names = info.get("TeamNames") or [a.get("Name") if isinstance(a, dict) else None for a in info.get("Agents", [])] or ["P0", "P1"]
    rewards = rep.get("rewards") or [None, None]
    n = len(steps)

    tl = [[], []]           # per player timeline
    events = [[], []]       # market fills
    prices = []             # per step market prices (state after step)
    mismatches = 0
    for t in range(1, n):
        prev, cur = steps[t - 1], steps[t]
        s = t - 1
        day, hour = s // TPD, s % TPD
        fills, money_sim = simulate_market(prev, cur, cfg)
        sh = _shared(cur)
        prices.append({"step": s, **sh["market"]["prices"]})
        for p in range(2):
            a = cur[p].get("action") or {}
            if not isinstance(a, dict):
                a = {}
            farm = sh["farms"][p]
            pv = _obs(cur, p).get("private") or {}
            rec = {
                "step": s, "day": day, "hour": hour,
                "farmer": act_str(a.get("farmer")),
                "hands": [act_str(h) for h in (a.get("hands") or [])],
                "market": a.get("market") or [],
                "money": farm["money"],
                "n_hands": len(_shared(prev)["farms"][p]["hands"]),
                "hands_after": len(farm["hands"]),
                "quadrants": len(farm["unlocked_quadrants"]),
            }
            if pv:
                rec["shed"] = {k: v for k, v in pv.get("shed", {}).items() if v}
                rec["seeds"] = {k: v for k, v in pv.get("seeds", {}).items() if v}
            if hour == TPD - 1 or t == n - 1 or t == 1:
                rec["tiles"] = tile_counts(farm["tiles"])
            tl[p].append(rec)
            if fills is not None:
                # money check: market sim vs recorded (end-of-day has no money effects)
                if abs(money_sim[p] - farm["money"]) > 0.5:
                    mismatches += 1
                for fr in fills[p]:
                    if fr["filled"] > 0 or fr["op"] in ("SELL", "BUY_PRODUCT", "BUY_LAND"):
                        events[p].append({"step": s, "day": day, "hour": hour, **fr,
                                          "total": sum(fr["prices"])})
    final_money = [steps[-1][0]["observation"]["farms"][p]["money"] for p in range(2)]
    return {
        "path": path, "names": names, "rewards": rewards, "final_money": final_money,
        "n_steps": n, "timeline": tl, "events": events, "prices": prices,
        "sim_mismatches": mismatches,
        "final_layout": [layout_str(steps[-1][0]["observation"]["farms"][p]["tiles"]) for p in range(2)],
        "shops": _shared(steps[-1])["town"]["unlocked_shops"],
        "shop_timeline": _shop_timeline(steps),
        "layouts": {d: [layout_str(_shared(steps[min(n - 1, (d + 1) * TPD)])["farms"][p]["tiles"]) for p in range(2)]
                    for d in (0, 1, 2, 5, 10, 15, 20, 25, 29)},
    }


def _shop_timeline(steps):
    out, last = [], 0
    for t in range(0, len(steps), TPD):
        shops = _shared(steps[t])["town"]["unlocked_shops"]
        if len(shops) > last:
            out.append((t // TPD, shops[-1]))
            last = len(shops)
    return out


# --------------------------------------------------------------- features
def features(parsed, p):
    tl = parsed["timeline"][p]
    ev = parsed["events"][p]
    oev = parsed["events"][1 - p]
    f = {"name": parsed["names"][p], "final_money": parsed["final_money"][p],
         "opp_name": parsed["names"][1 - p], "opp_final": parsed["final_money"][1 - p]}
    f["won"] = f["final_money"] > f["opp_final"]
    # opening: first day's per-turn actions (farmer + hands + market)
    f["opening_day0"] = [
        f"{r['hour']:02d} F:{r['farmer']} H:{r['hands']} M:{r['market']}" for r in tl if r["day"] == 0
    ]
    f["opening_d1_d2_market"] = [
        f"d{r['day']}h{r['hour']:02d} {r['market']}" for r in tl if 1 <= r["day"] <= 2 and r["market"]
    ]
    # purchases per day
    buys = defaultdict(Counter)
    for e in ev:
        if e["op"] in ("BUY_SEED", "BUY_ANIMAL", "BUY_PRODUCT") and e["filled"]:
            buys[e["day"]][e["item"]] += e["filled"]
    f["buys_by_day"] = {d: dict(c) for d, c in sorted(buys.items())}
    # plantings/placements per day (from farmer/hands PLANT/PLACE actions + tile counts at day end)
    f["tiles_by_day"] = {r["day"]: r["tiles"] for r in tl if "tiles" in r and r["hour"] == TPD - 1}
    f["land"] = [(e["day"], e["hour"], e["prices"][0] if e["prices"] else None) for e in ev
                 if e["op"] == "BUY_LAND" and e["filled"]]
    f["land_attempts_failed"] = sum(1 for e in ev if e["op"] == "BUY_LAND" and not e["filled"])
    hires = Counter()
    hire_cost = 0
    for e in ev:
        if e["op"] == "HIRE" and e["filled"]:
            hires[e["day"]] += 1
            hire_cost += e["total"]
    f["hires_by_day"] = dict(sorted(hires.items()))
    f["hire_cost_total"] = hire_cost
    sells = [e for e in ev if e["op"] == "SELL" and e["filled"]]
    f["sells"] = [(e["day"], e["hour"], e["item"], e["filled"], e["req"], e["prices"][0], e["prices"][-1],
                   round(e["total"] / e["filled"], 1)) for e in sells]
    rev = Counter()
    units = Counter()
    for e in sells:
        rev[e["item"]] += e["total"]
        units[e["item"]] += e["filled"]
    f["revenue"] = dict(rev)
    f["units_sold"] = dict(units)
    f["avg_price"] = {k: round(rev[k] / units[k], 1) for k in rev}
    f["buy_product"] = [(e["day"], e["hour"], e["item"], e["filled"], e["req"], e["total"]) for e in ev
                        if e["op"] == "BUY_PRODUCT"]
    spend = Counter()
    for e in ev:
        if e["op"] in ("BUY_SEED", "BUY_ANIMAL", "BUY_PRODUCT") and e["filled"]:
            spend[e["op"] + ":" + e["item"]] += e["total"]
    f["spend"] = dict(spend)
    # sells per hour-of-day histogram (timing policy)
    f["sell_hours"] = dict(sorted(Counter(e["hour"] for e in sells).items()))
    f["sell_days"] = dict(sorted(Counter(e["day"] for e in sells).items()))
    # money curve at day ends
    f["money_by_day"] = {r["day"]: r["money"] for r in tl if r["hour"] == TPD - 1}
    # end game: last 2 days
    f["endgame"] = [
        f"d{r['day']}h{r['hour']:02d} F:{r['farmer']} nH:{len(r['hands'])} M:{r['market']}"
        for r in tl if r["day"] >= 28 and (r["market"])
    ]
    f["end_shed"] = tl[-1].get("shed")
    f["final_tiles"] = tl[-1].get("tiles")
    # max hands concurrently
    f["max_hands"] = max((r["hands_after"] for r in tl), default=0)
    # farmer action mix
    mix = Counter()
    for r in tl:
        for a in [r["farmer"], *r["hands"]]:
            mix[a.split(" ")[0]] += 1
    f["action_mix"] = dict(mix.most_common())
    # opponent-reactive: sells of same item within +-1 turn of opponent sell
    osell = defaultdict(set)
    for e in oev:
        if e["op"] == "SELL" and e["filled"]:
            osell[e["item"]].add(e["step"])
    same = 0
    for e in sells:
        if any(abs(e["step"] - s) <= 1 for s in osell[e["item"]]):
            same += 1
    f["sells_near_opp_sell"] = same
    f["n_sell_orders"] = len(sells)
    # turn-0 wheat duel: wheat BUY_PRODUCT/SELL in first 2 steps
    f["t0_wheat"] = [(e["step"], e["op"], e["filled"], e["total"]) for e in ev
                     if e["step"] <= 1 and e["item"] == "WHEAT" and e["op"] in ("SELL", "BUY_PRODUCT")]
    return f


def one_line(parsed):
    a, b = parsed["final_money"]
    return (f"{os.path.basename(parsed['path'])}: {parsed['names'][0]}={a:.0f} vs "
            f"{parsed['names'][1]}={b:.0f} shops={parsed['shops']} mism={parsed['sim_mismatches']}")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--json")
    ap.add_argument("--print", action="store_true")
    ap.add_argument("--table", action="store_true")
    args = ap.parse_args()
    paths = sorted(glob.glob(os.path.join(args.path, "*.json"))) if os.path.isdir(args.path) else [args.path]
    out = []
    for pth in paths:
        try:
            pr = parse_replay(pth)
        except Exception as ex:  # noqa
            print("FAILED", pth, ex)
            continue
        if args.table:
            print(one_line(pr))
        feats = [features(pr, p) for p in range(2)]
        if args.print:
            for fe in feats:
                print("=" * 80)
                for k, v in fe.items():
                    if isinstance(v, list) and len(v) > 40:
                        v = v[:40] + ["..."]
                    print(f"{k}: {v}")
        out.append({"path": pth, "names": pr["names"], "final_money": pr["final_money"], "shops": pr["shops"],
                    "shop_timeline": pr["shop_timeline"], "layouts": pr["layouts"],
                    "sim_mismatches": pr["sim_mismatches"], "features": feats})
    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, default=str)


if __name__ == "__main__":
    main()
