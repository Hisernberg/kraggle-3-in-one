"""
AgriMind — a four-layer adaptive farming agent for Kaggriculture.
Kaggle submission entry point: agent(obs)
"""
from collections import Counter

CROPS = {
    "WHEAT":      {"seed": 10,  "first": 2,  "max_d": 4,  "max_y": 6, "ongoing": False},
    "CARROT":     {"seed": 20,  "first": 2,  "max_d": 3,  "max_y": 4, "ongoing": False},
    "TOMATO":     {"seed": 50,  "first": 8,  "max_d": 8,  "max_y": 4, "ongoing": True},
    "STRAWBERRY": {"seed": 100, "first": 10, "max_d": 10, "max_y": 4, "ongoing": True},
    "MELON":      {"seed": 80,  "first": 10, "max_d": 10, "max_y": 6, "ongoing": False},
}
ANIMALS = {
    "GOOSE": {"cost": 300, "first": 4, "interval": 1, "product": "EGG"},
    "COW":   {"cost": 400, "first": 8, "interval": 2, "product": "MILK"},
    "SHEEP": {"cost": 500, "first": 6, "interval": 3, "product": "WOOL"},
}
SHOP_DEMAND = {
    "BAKERY":         ["EGG", "WHEAT"],
    "PIZZA_SHOP":     ["MILK", "TOMATO", "WHEAT"],
    "BRUNCH_SPOT":    ["EGG", "WHEAT", "STRAWBERRY"],
    "YARN_STORE":     ["WOOL"],
    "ICE_CREAM_SHOP": ["STRAWBERRY", "MILK", "WHEAT"],
    "PET_CAFE":       ["CARROT"],
    "SMOOTHIE_SHOP":  ["STRAWBERRY", "MILK"],
    "FARMERS_MARKET": ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY"],
}
SELL_LIMITS = {"WHEAT":15,"CARROT":10,"TOMATO":6,"STRAWBERRY":4,"MELON":2,
               "EGG":10,"MILK":4,"WOOL":3,"FERTILIZER":10}


def _crop_roi(crop, price, days_left):
    c = CROPS[crop]
    ypd = (c["max_y"] / max(1, c["max_d"] + 4)) if c["ongoing"] else (c["max_y"] / max(1, c["max_d"]))
    return (ypd * price) / max(1, c["seed"])


def _animal_roi(animal, price, days_left):
    a = ANIMALS[animal]
    if days_left < a["first"] + 2:
        return 0.0
    return ((1.0 / a["interval"]) * price) / max(1, a["cost"] / 10)


def _scan_tiles(farm, day):
    rows = []
    for y in range(len(farm["tiles"])):
        for x in range(len(farm["tiles"])):
            t = farm["tiles"][y][x]
            if t is None or t == "LOCKED":
                continue
            info = {"x": x, "y": y, "tile": t}
            if isinstance(t, dict):
                kind = t.get("kind")
                if kind == "PLANT":
                    info.update({"type": "plant", "crop": t["crop"],
                                 "needs_water": not t.get("watered_today", False),
                                 "harvestable": t.get("yield_units", 0) > 0,
                                 "dying": t.get("consecutive_unwatered", 0) >= 1})
                elif kind in ("COOP", "PASTURE"):
                    if "animal" in t:
                        info.update({"type": "animal", "animal": t["animal"],
                                     "needs_feed": not t.get("fed_today", False),
                                     "fertilizer": t.get("fertilizer_available", False),
                                     "harvestable": t.get("yield_units", 0) > 0})
                    else:
                        info.update({"type": "structure"})
                elif kind == "WEED":
                    info.update({"type": "weed"})
            rows.append(info)
    return rows


def _manhattan(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])


def _priority(t, day):
    if t.get("type") == "plant":
        if t.get("dying"):       return 100
        if t.get("harvestable"): return 90
        if t.get("needs_water"): return 70
    if t.get("type") == "animal":
        if t.get("needs_feed"):  return 95
        if t.get("harvestable"): return 80
        if t.get("fertilizer"):  return 40
    if t.get("type") == "weed":
        return 30
    return 0


def _assign(workers, tiles, day):
    tiles = [t for t in tiles if _priority(t, day) > 0]
    tiles.sort(key=lambda t: -_priority(t, day))
    assignment, used = {}, set()
    for wi, wpos in enumerate(workers):
        best, best_score = None, -1
        for ti, t in enumerate(tiles):
            if ti in used: continue
            s = _priority(t, day) / (1 + _manhattan(wpos, (t["x"], t["y"])))
            if s > best_score:
                best_score, best = s, ti
        if best is not None:
            assignment[wi] = tiles[best]
            used.add(best)
        else:
            assignment[wi] = None
    return assignment


def _step_toward(pos, target):
    px, py = pos
    tx, ty = target
    if (px, py) == (tx, ty): return None
    if px < tx: return "EAST"
    if px > tx: return "WEST"
    if py < ty: return "SOUTH"
    if py > ty: return "NORTH"
    return None


def _plan_sales(shed, prices, day):
    orders = []
    for item, qty in (shed or {}).items():
        if qty <= 0: continue
        p = prices.get(item, 0)
        if p <= 1: continue
        if day >= 27:
            orders.append(["SELL", item, qty])
        else:
            orders.append(["SELL", item, min(qty, SELL_LIMITS.get(item, qty))])
    return orders


def _pick_target(shops, prices, cash, day):
    d = Counter()
    for s in shops or []:
        for p in SHOP_DEMAND.get(s, []):
            d[p] += 1
    dl = 30 - day
    best, best_score = None, -1.0
    for crop, c in CROPS.items():
        if c["seed"] > cash: continue
        price = (prices or {}).get(crop, 0)
        if price <= 0: continue
        s = _crop_roi(crop, price, dl) * (1 + 0.25 * d.get(crop, 0))
        if s > best_score: best, best_score = crop, s
    for a, x in ANIMALS.items():
        if x["cost"] > cash * 3: continue
        if dl < x["first"] + 4: continue
        price = (prices or {}).get(x.get("product"), 0)
        if price <= 0: continue
        s = _animal_roi(a, price, dl) * (1 + 0.25 * d.get(x.get("product"), 0))
        if s > best_score: best, best_score = a, s
    return best if best is not None else "WHEAT"


def _safe_int(v, default=0):
    try:    return int(v)
    except: return default


def agent(obs):
    obs  = obs or {}
    me   = _safe_int(obs.get("player", 0))
    day  = _safe_int(obs.get("day", 0))
    farms = obs.get("farms") or []
    farm  = farms[me] if me < len(farms) else {}
    priv  = obs.get("private") or {}
    mkt   = obs.get("market")  or {}
    town  = obs.get("town")    or {}

    prices = mkt.get("prices", {}) or {}
    cash   = farm.get("money", 0) or 0
    shops  = town.get("unlocked_shops", []) or []
    target = _pick_target(shops, prices, cash, day)

    shed  = priv.get("shed", {}) or {}
    seeds = priv.get("seeds", {}) or {}
    market_orders = _plan_sales(shed, prices, day)

    if target in CROPS:
        if seeds.get(target, 0) < 3 and cash >= CROPS[target]["seed"] * 3:
            market_orders.append(["BUY_SEED", target, 3])
    elif target in ANIMALS:
        if cash > ANIMALS[target]["cost"] * 2:
            market_orders.append(["BUY_ANIMAL", target, 1])

    tiles = _scan_tiles(farm, day) if farm else []

    farmer_pos = tuple(farm.get("farmer", [4, 4]))
    hands_pos  = [tuple(h) for h in (farm.get("hands", []) or [])]
    workers    = [farmer_pos] + hands_pos
    assignment = _assign(workers, tiles, day)

    def action_for(wi):
        t = assignment.get(wi)
        if t is None: return ["PASS"]
        pos = workers[wi]
        if tuple(pos) != (t["x"], t["y"]):
            mv = _step_toward(pos, (t["x"], t["y"]))
            return [mv] if mv else ["PASS"]
        if t["type"] == "plant":
            if t.get("dying") or t.get("needs_water"): return ["WATER"]
            if t.get("harvestable"):                    return ["HARVEST"]
        if t["type"] == "animal":
            if t.get("needs_feed"):  return ["FEED"]
            if t.get("harvestable"): return ["HARVEST"]
            if t.get("fertilizer"):  return ["COLLECT_FERTILIZER"]
        if t["type"] == "weed":
            return ["DIG"]
        if t["tile"] is None and seeds.get(target, 0) > 0:
            return ["PLANT", target]
        return ["PASS"]

    return {
        "farmer": action_for(0),
        "hands":  [action_for(i + 1) for i in range(len(hands_pos))],
        "market": market_orders[:10],
    }
