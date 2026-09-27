"""
Kaggriculture agent.

Strategy in one paragraph: labour is nearly free (a farm hand costs fib(n) coins
and returns 24 actions), so the binding constraints are TILES, WHEAT for animal
feed, and the market's willingness to absorb product. Eggs and wheat are
bottomless sinks (~$40 / ~$20 per unit no matter how much you dump); melon
(~$26k) and fertiliser (~$25k) are fixed pots that the first mover drains; milk
and wool are tiny pots that only pay off in proportion to the town shops that
happen to unlock. So: hire aggressively, put animals near the shed (highest
actions-per-tile) and crops further out, feed + care every animal every day
(care compounds into extra product per production tick), collect the free daily
fertiliser, and sell down a reserve-price curve computed from an exact replica
of the market model rather than dumping blindly.
"""

import math

# --------------------------------------------------------------------------
# Static game data -- mirrors kaggriculture.py exactly.
# --------------------------------------------------------------------------
CROPS = {
    "WHEAT":      {"seed": 10,  "first": 2,  "maxday": 4,  "interval": 0, "max": 6, "ongoing": False},
    "CARROT":     {"seed": 20,  "first": 2,  "maxday": 3,  "interval": 0, "max": 4, "ongoing": False},
    "TOMATO":     {"seed": 50,  "first": 8,  "maxday": 8,  "interval": 1, "max": 4, "ongoing": True},
    "STRAWBERRY": {"seed": 100, "first": 10, "maxday": 10, "interval": 2, "max": 4, "ongoing": True},
    "MELON":      {"seed": 80,  "first": 10, "maxday": 12, "interval": 0, "max": 6, "ongoing": False},
}

ANIMALS = {
    "GOOSE": {"cost": 300, "struct": "COOP",    "first": 4, "interval": 1, "held": 4, "prod": "EGG"},
    "COW":   {"cost": 400, "struct": "PASTURE", "first": 8, "interval": 2, "held": 6, "prod": "MILK"},
    "SHEEP": {"cost": 500, "struct": "PASTURE", "first": 6, "interval": 3, "held": 6, "prod": "WOOL"},
}

PRODUCTS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON",
            "EGG", "MILK", "WOOL", "FERTILIZER"]

MARKET_PARAMS = {
    "WHEAT":      {"base":  25, "I0": 10000, "T": 400, "bf": "sqrt",   "bt": 0.80, "af": "log",    "at": 0.20},
    "CARROT":     {"base":  35, "I0": 10000, "T": 450, "bf": "hinge",  "bt": 1.00, "af": "sqrt",   "at": 0.70},
    "TOMATO":     {"base":  60, "I0": 10000, "T": 200, "bf": "hinge",  "bt": 0.40, "af": "sqrt",   "at": 0.60},
    "STRAWBERRY": {"base": 120, "I0": 10000, "T": 100, "bf": "sqrt",   "bt": 0.70, "af": "linear", "at": 1.60},
    "MELON":      {"base": 250, "I0": 10000, "T": 300, "bf": "log",    "bt": 0.20, "af": "sq",     "at": 3.60},
    "EGG":        {"base":  50, "I0": 10000, "T": 332, "bf": "hinge",  "bt": 0.40, "af": "log",    "at": 0.20},
    "MILK":       {"base": 160, "I0": 10000, "T": 122, "bf": "sqrt",   "bt": 0.60, "af": "linear", "at": 1.60},
    "WOOL":       {"base": 200, "I0": 10000, "T": 105, "bf": "log",    "bt": 0.20, "af": "sq",     "at": 3.20},
    "FERTILIZER": {"base": 100, "I0": 10000, "T": 200, "bf": "linear", "bt": 0.40, "af": "linear", "at": 0.40},
}

HINGE_GAIN = 8.0
LAND_PRICES = [1000, 2000, 4000]
SHED_CAP = 100
MAX_HANDS = 15
# A task d steps away costs d travel actions plus the action itself, but travel
# amortises because a unit works a cluster once it arrives -- hence the 0.55.
# (An exponential discount makes the far quadrants invisible and they go to weed.)
TRAVEL_COST = 0.55
CLUSTER_WEIGHT = 0.12   # how much nearby pending work boosts a task
RACE_STRENGTH = 0.5     # how hard to undercut an opponent into a finite pot

# --- tuned policy knobs -------------------------------------------------
ANIMAL_TARGETS = {"SHEEP": 10, "COW": 12, "GOOSE": 34}
TOTAL_ANIMAL_CAP = 42
ANIMAL_ROI_MIN = 2.4        # an animal must return this multiple of its price
MELON_RESERVE_TILES = 14    # service capacity held back for the melon wave
SERVICE_FACTOR = 0.62       # share of a unit-day that is productive, not travel
HAND_DIVISOR = 18.0         # load units one hired hand absorbs per day
# Steady-state units per day once mature, including the CARE bonus, which banks
# 1/day and is paid out in full on each production tick (so a cared goose lays 2).
ANIMAL_RATE = {"GOOSE": 2.0, "COW": 1.5, "SHEEP": 1.33}
WHEAT_GROW_PRICE = 40   # above this, grow wheat instead of buying it
WHEAT_TILE_MAX = 16     # most tiles we will ever devote to growing feed
FERT_FIELD_PRICE = 45   # at or below this, spread fertiliser instead of selling it
MAX_EXTRA_QUADRANTS = 3 # land beyond this is bought but never worked
PLANT_COST = 1.9            # actions/day a planted tile costs, incl. travel
ANIMAL_COST = 4.2           # actions/day an animal tile costs, incl. travel

# Fraction of base price below which we stop selling (relaxed near the end of
# the season and when the shed is close to overflowing).
SELL_FLOOR_FRAC = {
    "WHEAT": 0.55, "CARROT": 0.48, "TOMATO": 0.45, "STRAWBERRY": 0.40,
    "MELON": 0.40, "EGG": 0.58, "MILK": 0.42, "WOOL": 0.42, "FERTILIZER": 0.28,
}


# --------------------------------------------------------------------------
# Market model (exact replica, so we can price our own orders before sending)
# --------------------------------------------------------------------------
def _shape(func, x, T):
    x = max(0.0, x)
    if func == "linear":
        return x
    if func == "sq":
        return x * x
    if func == "sqrt":
        return math.sqrt(x)
    if func == "log":
        return math.log(1.0 + x)
    if func == "log10":
        return math.log10(1.0 + x)
    if func == "hinge":
        if not T or T <= 0:
            return x
        u = x / T
        return u + HINGE_GAIN * max(0.0, u - 1.0) ** 2
    return x


def market_price(item, inventory):
    p = MARKET_PARAMS[item]
    base, I0, T = p["base"], p["I0"], p["T"]
    if inventory < I0:
        amp = p["bt"] * base / _shape(p["bf"], T, T)
        value = base + amp * _shape(p["bf"], I0 - inventory, T)
    else:
        amp = p["at"] * base / _shape(p["af"], T, T)
        value = base - amp * _shape(p["af"], inventory - I0, T)
    return max(1, int(round(value)))


def units_sellable(item, inventory, floor_price, cap):
    """How many units we can sell before the marginal price drops under floor."""
    n, inv = 0, inventory
    while n < cap:
        if market_price(item, inv) < floor_price:
            break
        n += 1
        inv += 1
    return n


def units_buyable(item, inventory, ceiling, cap, money):
    """How many units we can buy before the marginal price exceeds ceiling."""
    n, inv, spent = 0, inventory, 0
    while n < cap:
        unit_price = market_price(item, inv - 1)
        if unit_price > ceiling or spent + unit_price > money:
            break
        n += 1
        inv -= 1
        spent += unit_price
    return n


def sale_revenue(item, inventory, n):
    """Exact proceeds from dropping n units into the market from `inventory`."""
    total, inv = 0, int(inventory)
    for _ in range(int(max(0, min(n, 400)))):
        p = market_price(item, inv)
        total += p
        if p > 1:
            inv += 1
    return total


SHOPS = {
    "BAKERY":         ["EGG", "WHEAT"],
    "PIZZA_SHOP":     ["MILK", "TOMATO", "WHEAT"],
    "BRUNCH_SPOT":    ["EGG", "WHEAT", "STRAWBERRY"],
    "YARN_STORE":     ["WOOL"],
    "ICE_CREAM_SHOP": ["STRAWBERRY", "MILK", "WHEAT"],
    "PET_CAFE":       ["CARROT"],
    "SMOOTHIE_SHOP":  ["STRAWBERRY", "MILK"],
    "FARMERS_MARKET": ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY"],
}


def town_drain_per_day(item, unlocked_shops, turns_per_day=24,
                       shop_interval=4, center_interval=24):
    """Units of `item` the town removes from the market each day.

    This is what keeps milk and wool prices high: if the shops eat more than we
    produce, the pot never saturates and another cow is still worth buying.
    """
    per_shop_tick = turns_per_day // max(1, shop_interval)
    n = 0
    for s in unlocked_shops:
        products = SHOPS.get(s)
        if products and item in products:
            n += (2 if len(products) == 1 else 1) * per_shop_tick
    if item != "FERTILIZER":
        n += turns_per_day // max(1, center_interval)
    return n


def _fib(n):
    a, b = 1, 1
    for _ in range(n):
        a, b = b, a + b
    return a


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
def _g(obj, key, default=None):
    try:
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)
    except Exception:
        return default


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def shed_tiles(board):
    half = board // 2
    return [(half - 1, half - 1), (half, half - 1), (half - 1, half), (half, half)]


def natural_max(crop):
    """Peak yield reachable by watering alone (no fertiliser)."""
    cd = CROPS[crop]
    if cd["ongoing"]:
        return cd["max"]
    ws = (cd["maxday"] + 1) // 2
    return min(cd["max"], 1 + (cd["maxday"] - ws + 1))


def harvest_age(crop):
    cd = CROPS[crop]
    if cd["ongoing"]:
        return cd["first"]
    return (cd["maxday"] + 1) // 2 + natural_max(crop) - 2


def cycle_actions(crop):
    """Actions one tile costs over a full cycle: plant + waterings + harvest.

    Outside the yield window watering only matters for survival, and a plant
    survives on every-other-day watering, so those days are half price.
    """
    cd = CROPS[crop]
    ws = (cd["maxday"] + 1) // 2
    waterings = -(-ws // 2) + (natural_max(crop) - 1)
    return 2 + waterings


def is_animal(tile):
    return isinstance(tile, dict) and tile.get("animal") is not None


def is_plant(tile):
    return isinstance(tile, dict) and tile.get("kind") == "PLANT"


def is_weed(tile):
    return isinstance(tile, dict) and tile.get("kind") == "WEED"


def is_empty_struct(tile):
    return (isinstance(tile, dict)
            and tile.get("kind") in ("COOP", "PASTURE")
            and tile.get("animal") is None)


def _deposit_action(u, price, dumping):
    """Bank produce at the shed without throwing away carried feed.

    DROP empties the whole inventory in one action; PLACE moves a single item
    type. While a unit is still feeding animals its wheat is working stock, so
    PLACE the produce and keep the wheat. At the end of the game, dump the lot.
    """
    inv = u["inv"]
    if not dumping and inv.get("WHEAT", 0) > 0:
        best, best_val = None, 0.0
        for k in PRODUCTS:
            if k == "WHEAT":
                continue
            v = inv.get(k, 0) * price.get(k, 0)
            if v > best_val:
                best, best_val = k, v
        if best is not None:
            return ["PLACE", best, inv[best]]
    return ["DROP"]


def _step_toward(u, tx, ty):
    dx, dy = tx - u["x"], ty - u["y"]
    if abs(dx) >= abs(dy) and dx != 0:
        return ["EAST"] if dx > 0 else ["WEST"]
    if dy != 0:
        return ["SOUTH"] if dy > 0 else ["NORTH"]
    if dx != 0:
        return ["EAST"] if dx > 0 else ["WEST"]
    return ["PASS"]


# --------------------------------------------------------------------------
# Agent entry point
# --------------------------------------------------------------------------
def agent(obs, config=None):
    try:
        return _decide(obs, config)
    except Exception:
        return _fallback(obs)


def _fallback(obs):
    """Never crash and never forfeit: idle the crew but keep liquidating stock.

    A submission that throws scores whatever is already banked, so the cheapest
    insurance is to keep converting the shed into coins no matter what.
    """
    n, orders = 0, []
    try:
        farms = _g(obs, "farms", []) or []
        p = _g(obs, "player", 0) or 0
        n = len(farms[p].get("hands", []) or [])
    except Exception:
        n = 0
    try:
        shed = _g(_g(obs, "private", {}) or {}, "shed", {}) or {}
        for item in PRODUCTS:
            have = int(shed.get(item, 0) or 0)
            if have > 0:
                orders.append(["SELL", item, have])
    except Exception:
        orders = []
    return {"farmer": ["PASS"], "hands": [["PASS"]] * n, "market": orders[:10]}


def _decide(obs, config):
    # ---------------- state ----------------
    player = _g(obs, "player", 0) or 0
    farms = _g(obs, "farms", []) or []
    farm = farms[player]
    private = _g(obs, "private", {}) or {}
    shed = dict(_g(private, "shed", {}) or {})
    seeds = dict(_g(private, "seeds", {}) or {})
    invs = _g(private, "inventories", []) or []

    market = _g(obs, "market", {}) or {}
    minv = dict(_g(market, "inventory", {}) or {})
    price = dict(_g(market, "prices", {}) or {})
    for it in PRODUCTS:
        minv.setdefault(it, 10000)
        price.setdefault(it, MARKET_PARAMS[it]["base"])

    town = _g(obs, "town", {}) or {}
    shops = list(_g(town, "unlocked_shops", []) or [])

    tiles = farm["tiles"]
    board = len(tiles)
    turns_per_day = int(_g(config, "turnsPerDay", 24) or 24)
    episode_steps = int(_g(config, "episodeSteps", 720) or 720)
    total_days = max(1, episode_steps // turns_per_day)

    day = int(_g(obs, "day", 0) or 0)
    hour = int(_g(obs, "hour", 0) or 0)
    step = int(_g(obs, "step", day * turns_per_day + hour) or 0)
    last_day = total_days - 1
    days_left = last_day - day
    money = float(farm.get("money", 0))

    sheds = shed_tiles(board)

    def shed_dist(x, y):
        return min(abs(x - sx) + abs(y - sy) for sx, sy in sheds)

    # ---------------- survey the farm ----------------
    plants, animal_tiles, empty_structs, empties, weeds = [], [], [], [], []
    crop_counts = {}
    animal_counts = {"GOOSE": 0, "COW": 0, "SHEEP": 0}
    for y in range(board):
        for x in range(board):
            t = tiles[y][x]
            if t == "LOCKED":
                continue
            if t is None:
                empties.append((x, y))
            elif is_plant(t):
                plants.append((x, y, t))
                crop_counts[t["crop"]] = crop_counts.get(t["crop"], 0) + 1
            elif is_animal(t):
                animal_tiles.append((x, y, t))
                animal_counts[t["animal"]] = animal_counts.get(t["animal"], 0) + 1
            elif is_empty_struct(t):
                empty_structs.append((x, y, t))
            elif is_weed(t):
                weeds.append((x, y))
    empties.sort(key=lambda p: shed_dist(p[0], p[1]))
    n_animals = len(animal_tiles)
    shed_total = sum(shed.values())

    # ---------------- units ----------------
    hands = farm.get("hands", []) or []
    units = []
    for i in range(1 + len(hands)):
        pos = farm["farmer"] if i == 0 else hands[i - 1]
        inv = dict(invs[i]) if i < len(invs) and isinstance(invs[i], dict) else {}
        units.append({"i": i, "x": int(pos[0]), "y": int(pos[1]), "inv": inv})
    n_units = len(units)

    carried = {}
    for u in units:
        for k, v in u["inv"].items():
            carried[k] = carried.get(k, 0) + v

    endgame = day >= last_day
    dumping = endgame and hour >= turns_per_day - 10

    # ---------------- targets ----------------
    targets = ANIMAL_TARGETS
    total_animal_cap = TOTAL_ANIMAL_CAP
    # Growing wheat only beats buying it once the market price climbs.
    wheat_ratio = 0.35 if price["WHEAT"] < 34 else 0.85
    wheat_tile_target = _clamp(int(math.ceil(n_animals * wheat_ratio)), 3, WHEAT_TILE_MAX)
    wheat_have = shed.get("WHEAT", 0) + carried.get("WHEAT", 0)
    wheat_need_per_day = max(1, n_animals)

    # ---------------- hiring ----------------
    # A full crew of 10 costs 143 coins for the whole day (fib pricing), so the
    # only real limit is having work for them -- never gate this on being rich.
    load = n_animals * 3.2 + len(plants) * 1.3 + min(len(empties), 14) * 1.2
    if money < 120:
        hand_cap = 3
    elif money < 500:
        hand_cap = 6
    elif money < 1800:
        hand_cap = 9
    elif money < 6000:
        hand_cap = 11
    elif money < 16000:
        hand_cap = 13
    else:
        hand_cap = MAX_HANDS
    if days_left <= 0:
        hand_cap = min(hand_cap, 8)
    target_hands = _clamp(int(round(load * 1.6 / HAND_DIVISOR)), 2, hand_cap)
    hires_today = int(farm.get("hires_today", 0))
    hires_wanted = max(0, target_hands - hires_today) if hour <= 8 else 0

    # ---------------- how much work can the crew actually absorb? ----------------
    # Costs are in actions/day and include amortised travel. A plant that does
    # not get watered on its due day becomes a weed, so over-planting is not a
    # slow leak, it destroys the tile AND costs a DIG to recover.
    service_cap = (target_hands + 1) * turns_per_day * SERVICE_FACTOR
    current_load = len(plants) * PLANT_COST + n_animals * ANIMAL_COST
    plant_slots = max(0, int((service_cap - current_load) / PLANT_COST))
    room_to_plant = plant_slots > 0
    # Melon is ~$130/action, far above anything an animal returns, so hold
    # capacity back for the wave instead of letting the herd eat it all.
    melon_reserve = 0.0
    if days_left >= 11 and price["MELON"] >= 110:
        melon_reserve = PLANT_COST * max(0, MELON_RESERVE_TILES - crop_counts.get("MELON", 0))
    # Same logic for wheat once the market price climbs: at $55 wheat a home-grown
    # tile returns ~$37/action against ~$21 for a marginal goose, and not buying
    # also stops us bidding our own feed price up.
    wheat_reserve = 0.0
    if days_left >= 6 and price["WHEAT"] >= WHEAT_GROW_PRICE:
        wheat_reserve = PLANT_COST * max(0, wheat_tile_target - crop_counts.get("WHEAT", 0))
    room_for_animal = (current_load + melon_reserve + wheat_reserve
                       + ANIMAL_COST < service_cap)

    # ==========================================================
    # MARKET ORDERS  (max 10 per turn, processed in list order)
    # ==========================================================
    # Hold a few fertiliser units back for spreading on crops once the market
    # price has collapsed; below that price the field use is worth far more.
    fert_field_reserve = (8 if (price["FERTILIZER"] <= FERT_FIELD_PRICE
                                and not endgame and days_left >= 5) else 0)
    orders = []
    budget = money

    # 1. Hire -- by far the cheapest resource in the game. Only 10 market orders
    #    clear per turn, so leave slots for selling; we top the crew up on the
    #    next few hours anyway.
    hire_slots = 6 if shed_total > 70 else 8
    for k in range(min(hires_wanted, hire_slots)):
        cost = _fib(hires_today + k)
        if cost > max(30.0, budget * 0.25):
            break
        orders.append(["HIRE"])
        budget -= cost

    # 2. Emergency feed: losing a $400 cow beats any price we would pay for wheat.
    starving = (n_animals > 0 and wheat_have < wheat_need_per_day and not endgame)
    if starving:
        room = max(0, SHED_CAP - shed_total)
        want = min(room, wheat_need_per_day * 2 - wheat_have)
        n = units_buyable("WHEAT", minv["WHEAT"], 120, max(0, want), budget)
        if n > 0:
            orders.append(["BUY_PRODUCT", "WHEAT", n])
            budget -= n * price["WHEAT"]

    # 3. Sell down the reserve curve, biggest-value order first.
    # Unsold stock scores nothing, so the reserve price has to decay as the
    # season runs out -- holding out for $110 melon on day 27 just loses it all.
    relax = 1.0
    if shed_total > 62:
        relax *= 0.55
    if days_left <= 8:
        relax *= max(0.12, days_left / 8.0)
    if endgame:
        relax = 0.0

    # Race the opponent for the pots that never refill. Their farm is public, so
    # we can see the melons ripening and the animals dripping fertiliser. Whatever
    # they are about to dump, we want sold first -- second place into a saturated
    # pot gets the $1 floor.
    race = {}
    if RACE_STRENGTH > 0 and len(farms) > 1 and not endgame:
        opp_supply = {}
        for row in farms[1 - player]["tiles"]:
            for t in row:
                if is_animal(t):
                    a = t["animal"]
                    p_ = ANIMALS[a]["prod"]
                    opp_supply[p_] = opp_supply.get(p_, 0.0) + ANIMAL_RATE[a] * max(0, days_left)
                    opp_supply["FERTILIZER"] = opp_supply.get("FERTILIZER", 0.0) + max(0, days_left)
                elif is_plant(t):
                    c = t["crop"]
                    if not CROPS[c]["ongoing"]:
                        opp_supply[c] = opp_supply.get(c, 0.0) + natural_max(c)
        for item, supply in opp_supply.items():
            head = MARKET_PARAMS[item]["T"]          # scale by the resource's own size
            race[item] = max(0.40, 1.0 - RACE_STRENGTH * supply / float(head))

    sell_orders = []
    for item in PRODUCTS:
        avail = shed.get(item, 0)
        if item == "WHEAT" and not endgame:
            avail -= min(avail, int(wheat_need_per_day * 1.2))  # keep feed in reserve
        if item == "FERTILIZER" and fert_field_reserve:
            avail -= min(avail, fert_field_reserve)
        if avail <= 0:
            continue
        floor = max(1, int(MARKET_PARAMS[item]["base"] * SELL_FLOOR_FRAC[item]
                           * relax * race.get(item, 1.0)))
        n = units_sellable(item, minv[item], floor, avail)
        if n > 0:
            sell_orders.append((n * price[item], ["SELL", item, n]))
    sell_orders.sort(key=lambda kv: -kv[0])
    orders.extend(o for _, o in sell_orders)

    # 4. Land -- each quadrant is 25 more tiles, the single biggest multiplier.
    n_extra = len(farm.get("unlocked_quadrants", ["NW"])) - 1
    if (n_extra < min(len(LAND_PRICES), MAX_EXTRA_QUADRANTS) and days_left >= 10
            and len(empties) < 14 and room_for_animal):
        cost = LAND_PRICES[n_extra]
        if budget >= cost + 900:
            orders.append(["BUY_LAND"])
            budget -= cost

    # 5. Animals -- ranked by live return on investment, so a crashed wool price
    #    stops sheep purchases by itself.
    #    An animal we cannot feed escapes and burns its whole purchase price, so
    #    hold back roughly two days of bought feed for the existing herd first.
    in_transit = sum(shed.get(a, 0) + carried.get(a, 0) for a in ANIMALS)
    feed_float = 2.2 * (n_animals + in_transit + 1) * max(25, price["WHEAT"])
    if (day <= 21 and n_animals + in_transit < total_animal_cap
            and room_for_animal and in_transit <= 2):
        best, best_roi = None, 0.0
        for a in ("SHEEP", "COW", "GOOSE"):
            have = animal_counts.get(a, 0) + shed.get(a, 0) + carried.get(a, 0)
            if have >= targets[a]:
                continue
            prod = ANIMALS[a]["prod"]
            productive = max(0, days_left - ANIMALS[a]["first"] - 1)
            lifetime = ANIMAL_RATE[a] * productive
            # Price this animal against the market it will really sell into: the
            # rest of the herd dumps its output first, and the town drains some
            # back out. That is what stops sheep once wool saturates, and keeps
            # buying cows while ice-cream shops are still eating milk.
            glut = have * lifetime - town_drain_per_day(prod, shops) * max(0, days_left)
            start = minv[prod] + int(max(0, glut) * 0.5)
            revenue = sale_revenue(prod, start, lifetime)
            # Every surviving animal also yields one free fertiliser per day,
            # which is frequently worth more than the product itself.
            alive = max(0, days_left)
            fert_glut = n_animals * alive - town_drain_per_day("FERTILIZER", shops) * alive
            revenue += sale_revenue("FERTILIZER", minv["FERTILIZER"] + int(max(0, fert_glut) * 0.5), alive)
            # It eats one wheat every day it is alive, not just productive days.
            feed = alive * max(25, price["WHEAT"])
            roi = (revenue - feed) / float(ANIMALS[a]["cost"])
            if roi > best_roi:
                best, best_roi = a, roi
        if best and best_roi >= ANIMAL_ROI_MIN and shed_total < SHED_CAP - 2:
            cost = ANIMALS[best]["cost"]
            if budget >= cost + 250 + feed_float:
                orders.append(["BUY_ANIMAL", best, 1])
                budget -= cost

    # 6. Seeds (they live outside the shed, so stockpiling costs no capacity).
    # Melon has no town demand at all, so its ~158-unit pot never refills: only
    # plant as many tiles as the market will still absorb at a decent price.
    melon_room = units_sellable("MELON", minv["MELON"], 112, 400)
    melon_tile_cap = min(22, melon_room // natural_max("MELON"))
    melon_have = crop_counts.get("MELON", 0) + seeds.get("MELON", 0) + shed.get("MELON", 0) // 6
    melon_ok = (days_left >= 11 and price["MELON"] >= 110 and melon_have < melon_tile_cap)
    seed_wants = []
    if days_left >= 5:
        want = wheat_tile_target - crop_counts.get("WHEAT", 0) - seeds.get("WHEAT", 0)
        if want > 0:
            seed_wants.append(("WHEAT", min(want, 8)))
    if melon_ok:
        seed_wants.append(("MELON", min(melon_tile_cap - melon_have, 5)))
    if days_left >= 4 and price["CARROT"] >= 22:
        pending = seeds.get("CARROT", 0) + seeds.get("WHEAT", 0) + seeds.get("MELON", 0)
        want = len(empties) - pending
        if want > 0:
            seed_wants.append(("CARROT", min(want, 6)))
    for crop, qty in seed_wants:
        cost = CROPS[crop]["seed"] * qty
        if budget >= cost + 300:
            orders.append(["BUY_SEED", crop, qty])
            budget -= cost

    # 7. Top up wheat while it is cheap -- buying beats spending tiles on it.
    if not starving and n_animals > 0 and days_left >= 2:
        room = max(0, min(SHED_CAP - shed_total - 4, 34))
        want = max(0, int(wheat_need_per_day * 1.6) - wheat_have)
        if want > 0 and room > 0 and budget > 250:
            # An extra wheat turns into an extra product unit worth far more
            # than the wheat, so pay well above base before going hungry.
            n = units_buyable("WHEAT", minv["WHEAT"], 58, min(want, room), budget - 200)
            if n > 0:
                orders.append(["BUY_PRODUCT", "WHEAT", n])

    orders = orders[:10]

    # ==========================================================
    # TASKS
    # ==========================================================
    tasks = []

    def add(x, y, action, value, cap=1, need=None):
        tasks.append({"x": x, "y": y, "a": action, "v": float(value),
                      "cap": cap, "need": need, "taken": 0})

    # A weed is a dead tile; clearing one is worth roughly a crop cycle on it.
    weed_value = 28.0 if room_to_plant else 6.0
    fert_price = price["FERTILIZER"]
    # Using fertiliser on crops competes with selling it, so only stock up
    # for field use once the market has stopped paying well for it.
    fert_for_fields = fert_price <= FERT_FIELD_PRICE

    # --- plants ---
    for (x, y, t) in plants:
        crop = t["crop"]
        cd = CROPS[crop]
        age = day - int(t.get("planted_day", day))
        yu = int(t.get("yield_units", 0))
        nmax = natural_max(crop)
        p = price.get(crop, cd["seed"])
        ws = (cd["maxday"] + 1) // 2
        watered = bool(t.get("watered_today", False))
        dying = int(t.get("consecutive_unwatered", 0)) >= 1

        if not watered and not endgame:
            in_window = (not cd["ongoing"]) and ws <= age <= cd["maxday"] and yu < nmax
            v = 0.0
            if dying:
                # Miss this and we lose the crop, the tile, and an action to DIG.
                v = max(v, p * nmax * 0.9 + 35)
            if in_window:
                v = max(v, p + 4)                    # one extra unit of yield
            if cd["ongoing"] and dying:
                v = max(v, p * cd["max"] * 0.6)
            if v > 0:
                add(x, y, ["WATER"], v)

        # Fertilise at the start of the yield window: it doubles the per-day
        # watering bonus, lifting wheat from 4 units to its cap of 6. Worth doing
        # exactly when the extra produce beats what the fertiliser would fetch --
        # which is most of the late game, once our own animals have saturated the
        # fertiliser pot and driven its price into the floor.
        if (not cd["ongoing"]) and not endgame:
            ws_start = max(age, ws)
            if (ws - 1 <= age <= cd["maxday"] - 1
                    and int(t.get("fertilized_until_day", -1)) < day
                    and days_left >= cd["maxday"] - age):
                extra = min(cd["max"] - nmax, cd["maxday"] - ws_start + 1)
                gain = extra * p - fert_price
                if extra > 0 and gain > 15 and fert_for_fields:
                    add(x, y, ["FERTILIZE"], gain, need="FERTILIZER")

        if yu > 0 and age >= cd["first"]:
            mls = int(t.get("max_lifespan_step", -1))
            decaying = mls >= 0 and step >= mls - 1
            if cd["ongoing"]:
                ready = yu >= 2 or yu >= cd["max"] or endgame or decaying
            else:
                ready = yu >= nmax or age >= cd["maxday"] or decaying or endgame
            if ready:
                add(x, y, ["HARVEST"], yu * p + 6)

    # --- animals ---
    for (x, y, t) in animal_tiles:
        a = t["animal"]
        prod = ANIMALS[a]["prod"]
        pp = price.get(prod, 1)
        fed = bool(t.get("fed_today", False))
        cared = bool(t.get("cared_today", False))
        unfed = int(t.get("consecutive_unfed", 0))
        yu = int(t.get("yield_units", 0))

        if not fed and not endgame:
            v = pp * 1.3 + 10
            if unfed >= 1:
                v += ANIMALS[a]["cost"] * 0.9        # it escapes tonight otherwise
            add(x, y, ["FEED"], v, need="WHEAT")

        # CARE only banks a bonus if the animal also ends the day fed.
        if fed and not cared and days_left >= 1:
            add(x, y, ["CARE"], pp * 0.95 + 3)

        if t.get("fertilizer_available") and fert_price >= 6:
            add(x, y, ["COLLECT_FERTILIZER"], fert_price + 2)

        if yu > 0:
            held = ANIMALS[a]["held"]
            if yu >= 2 or yu >= held - 1 or endgame:
                add(x, y, ["HARVEST"], yu * pp + 5)

    # --- empty structures waiting for an animal ---
    for (x, y, t) in empty_structs:
        if t.get("kind") == "COOP":
            add(x, y, ["PLACE", "GOOSE"], 520, need="GOOSE")
        else:
            pick = "SHEEP" if carried.get("SHEEP", 0) >= carried.get("COW", 0) else "COW"
            if carried.get(pick, 0) == 0:
                pick = "COW" if carried.get("COW", 0) > 0 else "SHEEP"
            add(x, y, ["PLACE", pick], 560, need=pick)

    # --- weeds ---
    for (x, y) in weeds:
        add(x, y, ["DIG"], weed_value)

    # --- empty tiles: structures nearest the shed, crops further out ---
    pending = {a: shed.get(a, 0) + carried.get(a, 0) for a in ANIMALS}
    open_coops = sum(1 for (_, _, t) in empty_structs if t.get("kind") == "COOP")
    open_pastures = sum(1 for (_, _, t) in empty_structs if t.get("kind") == "PASTURE")
    need_coop = pending["GOOSE"] - open_coops
    need_pasture = (pending["COW"] + pending["SHEEP"]) - open_pastures
    # keep one spare coop ready so a purchase is never stalled waiting on a build
    if day <= 21 and n_animals + sum(pending.values()) < total_animal_cap:
        if targets["GOOSE"] > animal_counts.get("GOOSE", 0):
            need_coop += 1

    n_coop_slots = max(0, need_coop)
    build_slots = n_coop_slots + max(0, need_pasture)
    planted_here = 0
    # `empties` is sorted nearest-shed-first, so spending the plant budget in
    # order also keeps the worked footprint compact and cuts travel.
    for idx, (x, y) in enumerate(empties):
        if idx < build_slots and day <= 22:
            if idx < n_coop_slots:
                add(x, y, ["BUILD_COOP"], 240)
            else:
                add(x, y, ["BUILD_PASTURE"], 260)
            continue
        if planted_here >= plant_slots:
            continue
        best_crop, best_v = None, 0.0
        for crop in ("MELON", "WHEAT", "CARROT"):
            if seeds.get(crop, 0) <= 0:
                continue
            if crop == "MELON" and not melon_ok:
                continue
            if days_left < harvest_age(crop) + 1:
                continue
            # Coins per action over the whole cycle, so this is directly
            # comparable with feeding, harvesting and collecting fertiliser.
            v = ((natural_max(crop) * price.get(crop, 1) - CROPS[crop]["seed"])
                 / float(cycle_actions(crop)))
            if v > best_v:
                best_crop, best_v = crop, v
        if best_crop and best_v > 12:
            add(x, y, ["PLANT", best_crop], best_v)
            planted_here += 1

    # --- shed: pick up feed / animals, drop produce ---
    # Load a full day of feed in ONE trip. Grabbing 3 wheat at a time means a
    # walk back to the shed after every third animal, and that round trip costs
    # more actions than the feeding itself.
    to_feed = sum(1 for (_, _, t) in animal_tiles if not t.get("fed_today"))
    per_unit_wheat = _clamp(int(math.ceil(to_feed / float(max(1, n_units)))) + 3, 6, 16)
    for (sx, sy) in sheds:
        if shed.get("WHEAT", 0) > 0 and to_feed > 0 and not endgame:
            add(sx, sy, ["PICKUP", "WHEAT", per_unit_wheat],
                150, cap=max(1, n_units))
        if fert_for_fields and shed.get("FERTILIZER", 0) > 0 and crop_counts:
            add(sx, sy, ["PICKUP", "FERTILIZER", 4], 110, cap=max(1, n_units))
        for a in ("SHEEP", "COW", "GOOSE"):
            if shed.get(a, 0) > 0:
                add(sx, sy, ["PICKUP", a, 1], 480, cap=shed.get(a, 0))
        # Resolved per unit at emit time: PLACE banks one product and keeps the
        # feed wheat, DROP empties everything. Value is computed per unit below.
        tasks.append({"x": sx, "y": sy, "a": ["DROP"], "v": 0.0,
                      "cap": max(1, n_units), "need": None, "taken": 0,
                      "deposit": True})

    # ==========================================================
    # ASSIGNMENT  (greedy over unit x task, distance-discounted)
    # ==========================================================
    def unit_ok(u, t):
        need = t["need"]
        if need and u["inv"].get(need, 0) <= 0:
            return False
        a = t["a"]
        if a[0] == "PLACE" and u["inv"].get(a[1], 0) <= 0:
            return False
        if a[0] == "PICKUP":
            if sum(u["inv"].values()) > 60:
                return False
            # Do not re-collect what we are already carrying. Without this a unit
            # parked by the shed burns most of its day topping up wheat it has.
            if u["inv"].get(a[1], 0) >= (3 if a[1] == "WHEAT" else 1):
                return False
        return True

    def drop_value(u):
        inv = u["inv"]
        if dumping:
            val = sum(inv.get(k, 0) * price.get(k, 0) for k in PRODUCTS)
            return val * 1.5 + 40 if sum(inv.values()) > 0 else 0.0
        # Carried wheat is working stock for feeding, not surplus to bank. Counting
        # it here sends loaded feeders back to the shed to dump the very wheat they
        # just collected, and they then have to collect it again.
        val = sum(inv.get(k, 0) * price.get(k, 0) for k in PRODUCTS if k != "WHEAT")
        cnt = sum(v for k, v in inv.items() if k != "WHEAT")
        if cnt >= 8 or (shed_total < 55 and cnt >= 5):
            return val * 0.30
        return 0.0

    # Walking is over half of every unit-day, so prefer a task sitting in a dense
    # pocket of work: once a unit arrives it chains several jobs for free, while an
    # isolated task costs the whole walk for one action. Shed tasks are excluded
    # from the density map or every unit would be dragged back to the centre.
    density = {}
    for t in tasks:
        if t["a"][0] in ("PICKUP", "DROP"):
            continue
        key = (t["x"], t["y"])
        density[key] = density.get(key, 0.0) + t["v"]
    neighbourhood = {}
    for key in density:
        x0, y0 = key
        s = 0.0
        for dx in (-2, -1, 0, 1, 2):
            for dy in (-2, -1, 0, 1, 2):
                s += density.get((x0 + dx, y0 + dy), 0.0)
        neighbourhood[key] = s

    pairs = []
    for ui, u in enumerate(units):
        for ti, t in enumerate(tasks):
            if not unit_ok(u, t):
                continue
            v = t["v"]
            if t.get("deposit"):
                v = drop_value(u)
                if v <= 0:
                    continue
            else:
                nearby = neighbourhood.get((t["x"], t["y"]), v) - v
                v += CLUSTER_WEIGHT * max(0.0, nearby)
            d = abs(u["x"] - t["x"]) + abs(u["y"] - t["y"])
            pairs.append((v / (1.0 + TRAVEL_COST * d), ui, ti))
    pairs.sort(key=lambda z: -z[0])

    assigned, used = {}, set()
    for _, ui, ti in pairs:
        if ui in used:
            continue
        t = tasks[ti]
        if t["taken"] >= t["cap"]:
            continue
        assigned[ui] = ti
        t["taken"] += 1
        used.add(ui)
        if len(used) == n_units:
            break

    # ==========================================================
    # EMIT
    # ==========================================================
    plant_budget = dict(seeds)
    actions = [["PASS"] for _ in range(n_units)]
    for ui, u in enumerate(units):
        ti = assigned.get(ui)
        if ti is None:
            # Idle: drift back toward the shed so we are useful next turn.
            tx, ty = min(sheds, key=lambda s: abs(u["x"] - s[0]) + abs(u["y"] - s[1]))
            actions[ui] = _step_toward(u, tx, ty)
            continue
        t = tasks[ti]
        if u["x"] == t["x"] and u["y"] == t["y"]:
            a = list(t["a"])
            if t.get("deposit"):
                a = _deposit_action(u, price, dumping)
            if a[0] == "PLANT":
                # The engine drops ALL plant requests for a crop if they exceed
                # the seeds we hold, so never over-commit within one turn.
                crop = a[1]
                if plant_budget.get(crop, 0) <= 0:
                    actions[ui] = ["PASS"]
                    continue
                plant_budget[crop] -= 1
            actions[ui] = a
        else:
            actions[ui] = _step_toward(u, t["x"], t["y"])

    return {"farmer": actions[0] if actions else ["PASS"],
            "hands": actions[1:],
            "market": orders}
