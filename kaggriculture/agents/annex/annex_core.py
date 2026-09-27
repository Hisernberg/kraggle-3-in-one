"""SE-annex wrapper core (concatenated after a base agent's source by build_annex.py).

The base agent (renamed _base_agent) keeps running its farm exactly as before on a filtered observation
that hides the annex: the SE quadrant looks LOCKED, annex hands and their inventories are removed and
hires_today only counts the base's own hires. Once the base owns NE + SW and cash allows, the wrapper buys
SE and works it with its own extra hands (tomato / carrot / wheat chosen from town-shop demand), selling
annex produce on the town-consumption ticks.
"""
import copy as _ax_copy
import math as _ax_math

_AX_CROPS = {
    "WHEAT": {"seed": 10, "first": 2, "maxd": 4, "interval": 0, "maxy": 6, "ongoing": False},
    "CARROT": {"seed": 20, "first": 2, "maxd": 3, "interval": 0, "maxy": 4, "ongoing": False},
    "TOMATO": {"seed": 50, "first": 8, "maxd": 8, "interval": 1, "maxy": 4, "ongoing": True},
}
_AX_MARKET = {
    "WHEAT": (25, 400, "sqrt", 0.80, "log", 0.20), "CARROT": (35, 450, "hinge", 1.00, "sqrt", 0.70),
    "TOMATO": (60, 200, "hinge", 0.40, "sqrt", 0.60), "FERTILIZER": (100, 200, "linear", 0.40, "linear", 0.40),
}
_AX_SHOPS = {
    "BAKERY": ["EGG", "WHEAT"], "PIZZA_SHOP": ["MILK", "TOMATO", "WHEAT"],
    "BRUNCH_SPOT": ["EGG", "WHEAT", "STRAWBERRY"], "YARN_STORE": ["WOOL"],
    "ICE_CREAM_SHOP": ["STRAWBERRY", "MILK", "WHEAT"], "PET_CAFE": ["CARROT"],
    "SMOOTHIE_SHOP": ["STRAWBERRY", "MILK"], "FARMERS_MARKET": ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY"],
}
_AX_SHED = (5, 5)
_AX_LAST = 718

AX = {
    "start_day": 9,          # earliest day to buy SE
    "cash_buffer": 1500,     # cash that must remain after buying SE
    "hands": 3,              # annex crew size
    "tomato_last_plant": 18,
    "carrot_last_plant": 26,
    "wheat_last_plant": 25,
    "fert_max_price": 70,
}


def _ax_shape(f, x, T):
    x = max(0.0, x)
    if f == "linear":
        return x
    if f == "sq":
        return x * x
    if f == "sqrt":
        return _ax_math.sqrt(x)
    if f == "log":
        return _ax_math.log(1.0 + x)
    if f == "hinge":
        u = x / T
        return u + 8.0 * max(0.0, u - 1.0) ** 2
    return x


def _ax_price(item, inv):
    base, T, bf, bt, af, at = _AX_MARKET[item]
    if inv < 10000:
        p = base + bt * base / _ax_shape(bf, T, T) * _ax_shape(bf, 10000 - inv, T)
    else:
        p = base - at * base / _ax_shape(af, T, T) * _ax_shape(af, inv - 10000, T)
    return max(1, int(round(p)))


def _ax_fib(n):
    a, b = 1, 1
    for _ in range(n):
        a, b = b, a + b
    return a


def _ax_g(o, k, d=None):
    return o.get(k, d) if isinstance(o, dict) else getattr(o, k, d)


def _ax_dist(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _ax_toward(p, q):
    if p[0] < q[0]:
        return "EAST"
    if p[0] > q[0]:
        return "WEST"
    if p[1] < q[1]:
        return "SOUTH"
    if p[1] > q[1]:
        return "NORTH"
    return "PASS"


def _ax_is_se(x, y):
    return x >= 5 and y >= 5


class _AxMem:
    def __init__(self):
        self.last_step = -1
        self.day = -1
        self.owner = []            # per hand index: "B" base / "A" annex (today)
        self.pending = []          # owners of HIRE orders submitted last turn, in order
        self.base_hires = 0
        self.own_se = False
        self.crop_of = {}          # (x,y) -> crop we planted
        self.planted_day = {}
        self.fert_reserve = 0


_AXM = _AxMem()


def _ax_demand(shops):
    d = {"WHEAT": 1.0, "CARROT": 1.0, "TOMATO": 1.0}
    for s in shops:
        prods = _AX_SHOPS.get(s, [])
        m = 2 if len(prods) == 1 else 1
        for p in prods:
            if p in d:
                d[p] += 6 * m
    return d


def _ax_base_view(obs, me, annex_idx, hide_se, hide_items=None):
    o = _ax_copy.deepcopy(obs) if not isinstance(obs, dict) else _ax_copy.deepcopy(dict(obs))
    farms = o["farms"]
    f = farms[me]
    keep = [i for i in range(len(f["hands"])) if i not in annex_idx]
    f["hands"] = [f["hands"][i] for i in keep]
    invs = o["private"]["inventories"]
    o["private"]["inventories"] = [invs[0]] + [invs[i + 1] for i in keep if i + 1 < len(invs)]
    f["hires_today"] = _AXM.base_hires
    for k, v in (hide_items or {}).items():
        o["private"]["shed"][k] = max(0, o["private"]["shed"].get(k, 0) - v)
    if hide_se:
        f["unlocked_quadrants"] = [q for q in f["unlocked_quadrants"] if q != "SE"]
        for y in range(5, 10):
            for x in range(5, 10):
                f["tiles"][y][x] = "LOCKED"
    return o


def _ax_annex_actions(obs, me, units, invs, money):
    """Return (unit actions for annex units, market orders) for the SE annex."""
    step = _ax_g(obs, "step")
    day, hour = step // 24, step % 24
    f = obs["farms"][me]
    tiles = f["tiles"]
    shed = dict(obs["private"]["shed"])
    seeds = dict(obs["private"]["seeds"])
    prices = obs["market"]["prices"]
    minv = obs["market"]["inventory"]
    shops = obs["town"]["unlocked_shops"]
    dem = _ax_demand(shops)
    orders = []
    final = step >= _AX_LAST - 5
    # ---- tasks on SE tiles
    tasks = {}
    empties = []
    fert_have = sum(i.get("FERTILIZER", 0) for i in invs) + shed.get("FERTILIZER", 0)
    for y in range(5, 10):
        for x in range(5, 10):
            t = tiles[y][x]
            if t is None:
                empties.append((x, y))
                continue
            if not isinstance(t, dict):
                continue
            k = t.get("kind")
            if k == "WEED":
                if day <= 27:
                    tasks[(x, y)] = [("DIG", None, None)]
                continue
            if k != "PLANT":
                continue
            c = t["crop"]
            cd = _AX_CROPS.get(c)
            if cd is None:
                continue
            age = day - t["planted_day"]
            yu = t.get("yield_units", 0)
            must = t.get("consecutive_unwatered", 0) >= 1
            decaying = t["max_lifespan_step"] >= 0 and step >= t["max_lifespan_step"]
            ops = []
            fert_age = (c in ("WHEAT", "CARROT") and age == 2) or (c == "TOMATO" and age in (7, 10))
            if fert_age and t.get("fertilized_until_day", -1) < day and not t["watered_today"] and fert_have > 0:
                ops.append(("FERTILIZE", None, "FERTILIZER"))
                fert_have -= 1
            if not cd["ongoing"]:
                ready = age >= cd["first"] and (yu >= cd["maxy"] or decaying or (age >= cd["maxd"] and t["watered_today"])
                                                or (step >= _AX_LAST - 30 and t["watered_today"]))
                in_window = (cd["maxd"] + 1) // 2 <= age <= cd["maxd"] and yu < cd["maxy"]
                if ready:
                    ops.append(("HARVEST", None, None))
                elif not t["watered_today"] and (must or in_window):
                    ops.append(("WATER", None, None))
            else:
                dsf = day + 1 - t["planted_day"] - cd["first"]
                prod = dsf >= 0 and dsf % cd["interval"] == 0 and dsf // cd["interval"] + 1 <= cd["maxy"]
                fert_now = t.get("fertilized_until_day", -1) >= day
                if not t["watered_today"] and not decaying and (must or (prod and fert_now)):
                    ops.append(("WATER", None, None))
                if yu > 0 and (yu >= 2 or decaying or day >= 28):
                    ops.append(("HARVEST", None, None))
                elif decaying and yu <= 0:
                    ops.append(("DIG", None, None))
            if ops:
                tasks[(x, y)] = ops
    # ---- planting plan
    plant_jobs = []
    if not final and hour <= 20:
        for e in sorted(empties, key=lambda p: _ax_dist(p, _AX_SHED)):
            c = None
            if day <= AX["tomato_last_plant"] and dem["TOMATO"] >= 7:
                c = "TOMATO"
            elif day <= AX["carrot_last_plant"] and dem["CARROT"] >= dem["WHEAT"] * 0.5 and dem["CARROT"] >= 7:
                c = "CARROT"
            elif day <= AX["wheat_last_plant"]:
                c = "WHEAT"
            elif day <= AX["carrot_last_plant"]:
                c = "CARROT"
            if c:
                plant_jobs.append((e, c))
    # ---- selling annex crops (tick aligned, price guarded, liquidate at the end)
    tick = step % 4 == 1
    for item in ("TOMATO", "CARROT"):
        have = shed.get(item, 0)
        if have <= 0:
            continue
        base = _AX_MARKET[item][0]
        days_left = (_AX_LAST - step) / 24.0
        if final:
            n = have
        else:
            if not tick and days_left > 1.0:
                continue
            thr = base * (0.9 if days_left > 3 else 0.5 if days_left > 1 else 0.1)
            lot = 5 if days_left > 1 else have
            n = 0
            inv = minv[item]
            while n < min(have, lot) and _ax_price(item, inv + n) >= thr:
                n += 1
        if n > 0:
            orders.append(["SELL", item, int(n)])
            money += n * prices.get(item, 1) * 0.9
    # ---- seeds just in time (bought now, planted next turn)
    need = {}
    for (e, c) in plant_jobs:
        need[c] = need.get(c, 0) + 1
    for c, n in need.items():
        n = min(n - seeds.get(c, 0), len(units) + 1)
        cost = _AX_CROPS[c]["seed"]
        n = int(min(n, max(0, (money - 800) // cost)))
        if n > 0:
            orders.append(["BUY_SEED", c, n])
            money -= n * cost
    # ---- fertilizer purchases for tomatoes/carrots
    fert_need = sum(1 for ops in tasks.values() for o in ops if o[0] == "FERTILIZE")
    fert_need += sum(1 for (x, y) in [(x, y) for y in range(5, 10) for x in range(5, 10)]
                     if isinstance(tiles[y][x], dict) and tiles[y][x].get("kind") == "PLANT"
                     and tiles[y][x]["crop"] == "TOMATO" and day - tiles[y][x]["planted_day"] in (6, 9))
    _AXM.fert_reserve = min(shed.get("FERTILIZER", 0), fert_need)
    if fert_need > fert_have and prices.get("FERTILIZER", 100) <= AX["fert_max_price"] and money > 1500:
        k = int(min(fert_need - fert_have, 6))
        if k > 0:
            orders.append(["BUY_PRODUCT", "FERTILIZER", k])
            money -= k * prices.get("FERTILIZER", 100)
            _AXM.fert_reserve += k
    # ---- assign annex units
    acts = []
    taken = set()
    seeds_used = {}
    shed_fert = shed.get("FERTILIZER", 0)
    for idx, pos in enumerate(units):
        pos = tuple(pos)
        inv = invs[idx]
        produce = sum(v for k, v in inv.items() if k not in ("FERTILIZER",))
        if final:
            acts.append(["DROP"] if (produce or inv.get("FERTILIZER")) and pos == _AX_SHED else
                        [_ax_toward(pos, _AX_SHED)] if produce else ["PASS"])
            continue
        if pos == _AX_SHED and produce and sum(shed.values()) < 97:
            acts.append(["DROP"])
            continue
        best = None
        for tpos, ops in list(tasks.items()) + [((e), [("PLANT", c, "SEED")]) for (e, c) in plant_jobs]:
            if tpos in taken:
                continue
            op = ops[0]
            detour = 0
            if op[2] == "SEED":
                if seeds.get(op[1], 0) - seeds_used.get(op[1], 0) <= 0:
                    continue
            elif op[2] == "FERTILIZER" and inv.get("FERTILIZER", 0) <= 0:
                if shed_fert <= 0:
                    if len(ops) > 1:
                        op = ops[1]
                    else:
                        continue
                else:
                    detour = _ax_dist(pos, _AX_SHED) + 1 + _ax_dist(_AX_SHED, tpos) - _ax_dist(pos, tpos)
            d = _ax_dist(pos, tpos) + detour
            if best is None or d < best[0]:
                best = (d, tpos, op, detour)
        if best is None:
            acts.append([_ax_toward(pos, _AX_SHED)] if produce else ["PASS"])
            continue
        _, tpos, op, detour = best
        taken.add(tpos)
        if detour > 0:
            if pos == _AX_SHED:
                k = max(1, min(shed_fert, 4))
                shed_fert -= k
                acts.append(["PICKUP", "FERTILIZER", int(k)])
            else:
                acts.append([_ax_toward(pos, _AX_SHED)])
            continue
        if pos != tpos:
            acts.append([_ax_toward(pos, tpos)])
            continue
        if op[0] == "PLANT":
            seeds_used[op[1]] = seeds_used.get(op[1], 0) + 1
            acts.append(["PLANT", op[1]])
        else:
            acts.append([op[0]])
    return acts, orders, len(tasks) + len(plant_jobs)


def annex_agent_entry(obs, config=None):
    step = _ax_g(obs, "step", None)
    if step is None:
        step = _ax_g(obs, "day", 0) * 24 + _ax_g(obs, "hour", 0)
    global _AXM
    if step == 0 or step < _AXM.last_step:
        _AXM = _AxMem()
    M = _AXM
    M.last_step = step
    day, hour = step // 24, step % 24
    me = _ax_g(obs, "player")
    f = obs["farms"][me]
    try:
        if day != M.day:
            M.day = day
            M.owner = []
            M.pending = []
            M.base_hires = 0
        # resolve last turn's hires: new hands appear in order of the HIRE orders submitted
        n_h = len(f["hands"])
        new = n_h - len(M.owner)
        for k in range(max(0, new)):
            who = M.pending[k] if k < len(M.pending) else "B"
            M.owner.append(who)
            if who == "B":
                M.base_hires += 1
        M.pending = []
        annex_idx = [i for i, w in enumerate(M.owner) if w == "A"]
        M.own_se = "SE" in f["unlocked_quadrants"]
        hide = {}
        if M.own_se:
            hide["FERTILIZER"] = M.fert_reserve
            for item in ("TOMATO",):
                hide[item] = obs["private"]["shed"].get(item, 0)
        bobs = _ax_base_view(obs, me, set(annex_idx), M.own_se, hide)
        base = _base_agent(bobs, config)
        if not isinstance(base, dict):
            base = {"farmer": ["PASS"], "hands": [], "market": []}
        bmarket = [list(o) for o in (base.get("market") or [])]
        if M.own_se:
            bmarket = [o for o in bmarket if o and o[0] != "BUY_LAND"]
        base_hands = list(base.get("hands") or [])
        money = float(f["money"])
        orders = []
        # ---- annex purchase of SE
        quads = f["unlocked_quadrants"]
        if not M.own_se and len(quads) == 3 and "SW" in quads and day >= AX["start_day"] and day <= 16:
            if money - 4000 >= AX["cash_buffer"] and not any(o and o[0] == "BUY_LAND" for o in bmarket):
                orders.append(["BUY_LAND"])
        annex_actions = []
        if M.own_se:
            units = [f["hands"][i] for i in annex_idx]
            invs = [obs["private"]["inventories"][i + 1] if i + 1 < len(obs["private"]["inventories"]) else {}
                    for i in annex_idx]
            annex_actions, aorders, nwork = _ax_annex_actions(obs, me, units, invs, money)
            orders += aorders
            # hire the annex crew for the day (after the base's own hires)
            want = AX["hands"] if (nwork > 0 and day <= 28 and hour <= 14) else 0
            have = len(annex_idx)
            if want > have:
                for k in range(want - have):
                    orders.append(["HIRE"])
        # ---- merge
        market = bmarket + orders
        market = market[:10]
        M.pending = [("B" if j < len(bmarket) else "A") for j, o in enumerate(market) if o and o[0] == "HIRE"]
        hands = []
        bi = 0
        ai = 0
        for i, w in enumerate(M.owner):
            if w == "A":
                hands.append(annex_actions[ai] if ai < len(annex_actions) else ["PASS"])
                ai += 1
            else:
                hands.append(base_hands[bi] if bi < len(base_hands) else ["PASS"])
                bi += 1
        return {"farmer": base.get("farmer") or ["PASS"], "hands": hands, "market": market}
    except Exception:
        try:
            return _base_agent(obs, config)
        except Exception:
            return {"farmer": ["PASS"], "hands": [], "market": []}
