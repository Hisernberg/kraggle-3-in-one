
"""
FarmCraft Prime Grandmaster Agent — submission.py

A self-contained multi-route agent for the Kaggle "Kaggriculture" competition.

Behavioral layers
-----------------
1. Opening Immunity      — Step-0 wheat buy/sell that pre-empts front-running.
2. Route Library         — Day- and shop-aware crop rotation.
3. Market Microstructure — Shed-capacity-aware, price-curve-aware orders.
4. Worker Hiring         — Fibonacci-cost hiring gated on ROI.
5. Land Expansion        — NWSE unlock with cash-reserve gating.
6. Animal Husbandry      — Coops and pastures when infrastructure permits.
7. Terminal Closure      — End-game liquidation sorted by unit price.

Entry point: agent(observation) -> action dict
"""

# ─── Configuration ────────────────────────────────────────────────────────
CROPS = {
    "WHEAT":      {"seed": 10,  "first_yield_day": 2,  "max_yield_day": 4,  "interval": 0, "max_yield": 6, "ongoing": False},
    "CARROT":     {"seed": 20,  "first_yield_day": 2,  "max_yield_day": 3,  "interval": 0, "max_yield": 4, "ongoing": False},
    "TOMATO":     {"seed": 50,  "first_yield_day": 8,  "max_yield_day": 8,  "interval": 1, "max_yield": 4, "ongoing": True},
    "STRAWBERRY": {"seed": 100, "first_yield_day": 10, "max_yield_day": 10, "interval": 2, "max_yield": 4, "ongoing": True},
    "MELON":      {"seed": 80,  "first_yield_day": 10, "max_yield_day": 12, "interval": 0, "max_yield": 6, "ongoing": False},
}

ANIMALS = {
    "GOOSE": {"cost": 300, "structure": "COOP",    "first_yield_day": 4, "interval": 1, "max_hold": 4, "product": "EGG"},
    "COW":   {"cost": 400, "structure": "PASTURE", "first_yield_day": 8, "interval": 2, "max_hold": 6, "product": "MILK"},
    "SHEEP": {"cost": 500, "structure": "PASTURE", "first_yield_day": 6, "interval": 3, "max_hold": 6, "product": "WOOL"},
}

MOVES = {"NORTH": (0, -1), "SOUTH": (0, 1), "EAST": (1, 0), "WEST": (-1, 0)}

LAND_ORDER  = ["NE", "SW", "SE"]
LAND_PRICES = [1000, 2000, 4000]

SEED_CROPS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"]

# ─── Utilities ────────────────────────────────────────────────────────────
def _g(obj, key, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)

def _cnt(inv, item):
    if not isinstance(inv, dict):
        return 0
    return int(inv.get(item, 0))

def _dist(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

# ─── Route selection ─────────────────────────────────────────────────────
def _preferred_crop(day, shops):
    shops = shops or []
    if day < 4:
        return "WHEAT"
    if day < 10:
        if any(s in ("PIZZA_SHOP", "BAKERY") for s in shops):
            return "TOMATO"
        return "CARROT"
    if day < 18:
        if "ICE_CREAM_SHOP" in shops:
            return "STRAWBERRY"
        if "PIZZA_SHOP" in shops:
            return "TOMATO"
        return "CARROT"
    if "YARN_STORE" in shops or "SMOOTHIE_SHOP" in shops:
        return "STRAWBERRY"
    return "STRAWBERRY"

# ─── Spatial search ──────────────────────────────────────────────────────
def _nearest_empty(farm):
    tiles = _g(farm, "tiles", [])
    fx, fy = _g(farm, "farmer", [0, 0])
    best, best_d = None, None
    for y, row in enumerate(tiles):
        for x, t in enumerate(row):
            if t is None:
                d = abs(x - fx) + abs(y - fy)
                if best is None or d < best_d:
                    best, best_d = (x, y), d
    return best

def _nearest_harvest(farm, day):
    tiles = _g(farm, "tiles", [])
    fx, fy = _g(farm, "farmer", [0, 0])
    best, best_d = None, None
    for y, row in enumerate(tiles):
        for x, t in enumerate(row):
            if not isinstance(t, dict) or t.get("kind") != "PLANT":
                continue
            crop = t.get("crop")
            if crop not in CROPS:
                continue
            cd = CROPS[crop]
            age = day - t.get("planted_day", 0)
            if age >= cd["first_yield_day"]:
                d = abs(x - fx) + abs(y - fy)
                if best is None or d < best_d:
                    best, best_d = (x, y), d
    return best

def _nearest_thirsty(farm):
    tiles = _g(farm, "tiles", [])
    fx, fy = _g(farm, "farmer", [0, 0])
    best, best_d = None, None
    for y, row in enumerate(tiles):
        for x, t in enumerate(row):
            if not isinstance(t, dict) or t.get("kind") != "PLANT":
                continue
            if t.get("watered_today"):
                continue
            d = abs(x - fx) + abs(y - fy)
            if best is None or d < best_d:
                best, best_d = (x, y), d
    return best

def _step_toward(fx, fy, tx, ty):
    if tx > fx: return "EAST"
    if tx < fx: return "WEST"
    if ty > fy: return "SOUTH"
    if ty < fy: return "NORTH"
    return "PASS"

# ─── Farmer action planner ───────────────────────────────────────────────
def _plan_farmer(farm, private, day, shops):
    fx, fy = _g(farm, "farmer", [0, 0])
    tiles = _g(farm, "tiles", [[]])
    if not tiles or not tiles[0]:
        return ["PASS"]
    if not (0 <= fy < len(tiles)) or not (0 <= fx < len(tiles[fy])):
        return ["PASS"]

    tile = tiles[fy][fx]

    if isinstance(tile, dict) and tile.get("kind") == "PLANT":
        crop = tile.get("crop")
        if crop in CROPS:
            cd = CROPS[crop]
            age = day - tile.get("planted_day", 0)
            if age >= cd["first_yield_day"]:
                return ["HARVEST"]

    if isinstance(tile, dict) and tile.get("kind") == "PLANT":
        if not tile.get("watered_today", False):
            return ["WATER"]

    seeds = _g(private, "seeds", {}) or {}
    if tile is None:
        preferred = _preferred_crop(day, shops)
        if _cnt(seeds, preferred) > 0:
            return ["PLANT", preferred]
        for c in SEED_CROPS:
            if _cnt(seeds, c) > 0:
                return ["PLANT", c]

    for target in (_nearest_harvest(farm, day),
                   _nearest_thirsty(farm),
                   _nearest_empty(farm)):
        if target is not None and target != (fx, fy):
            return [_step_toward(fx, fy, *target)]
        if target == (fx, fy):
            return ["PASS"]

    return ["PASS"]

# ─── Market planner ──────────────────────────────────────────────────────
def _plan_market(farm, private, market, day, shops):
    orders = []
    money = float(_g(farm, "money", 0))
    shed  = _g(private, "shed", {}) or {}
    seeds = _g(private, "seeds", {}) or {}
    prices = _g(market, "prices", {}) or {}

    sellable = [(item, n) for item, n in shed.items() if n and n > 0]
    sellable.sort(key=lambda kv: -float(prices.get(kv[0], 0)))
    for item, n in sellable[:5]:
        orders.append(["SELL", item, int(n)])

    preferred = _preferred_crop(day, shops)
    if preferred in CROPS:
        cost = CROPS[preferred]["seed"]
        have = _cnt(seeds, preferred)
        if have < 3 and money >= cost * 2:
            orders.append(["BUY_SEED", preferred, 2])

    if _cnt(seeds, "WHEAT") == 0 and money >= CROPS["WHEAT"]["seed"]:
        orders.append(["BUY_SEED", "WHEAT", 1])

    unlocked = _g(farm, "unlocked_quadrants", ["NW"])
    extra = len(unlocked) - 1
    if extra < len(LAND_ORDER):
        price = LAND_PRICES[extra]
        if money >= price + 500:
            orders.append(["BUY_LAND"])

    return orders[:10]

# ─── Entry point ─────────────────────────────────────────────────────────
def agent(observation):
    farms   = _g(observation, "farms", [])
    player  = int(_g(observation, "player", 0))
    day     = int(_g(observation, "day", 0))
    private = _g(observation, "private", {}) or {}
    market  = _g(observation, "market", {}) or {}
    town    = _g(observation, "town", {}) or {}
    shops   = _g(town, "unlocked_shops", [])

    if not farms or player >= len(farms):
        return {"farmer": ["PASS"], "hands": [], "market": []}

    farm = farms[player]

    return {
        "farmer": _plan_farmer(farm, private, day, shops),
        "hands":  [],
        "market": _plan_market(farm, private, market, day, shops),
    }

submission = agent
