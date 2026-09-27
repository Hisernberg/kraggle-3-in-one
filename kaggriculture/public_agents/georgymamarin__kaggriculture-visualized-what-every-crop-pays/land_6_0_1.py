
CARROT, MAX_YIELD_DAY = "CARROT", 3
PATCH, CREW, BUY_LAND = [(4, 4), (3, 4), (4, 3), (2, 4), (3, 3), (4, 2)], 0, False
SHED = (4, 4)

def step_toward(pos, target):
    (x, y), (tx, ty) = pos, target
    if x < tx: return ["EAST"]
    if x > tx: return ["WEST"]
    if y < ty: return ["SOUTH"]
    if y > ty: return ["NORTH"]
    return ["PASS"]

def tile_needs(tile, seeds, day):
    if isinstance(tile, dict) and tile.get("kind") == "WEED": return ["DIG"]
    if tile is None:
        return ["PLANT", CARROT] if seeds.get(CARROT, 0) > 0 else None
    if isinstance(tile, dict) and tile.get("kind") == "PLANT":
        if not tile.get("watered_today"): return ["WATER"]
        if day - tile["planted_day"] >= MAX_YIELD_DAY and tile.get("yield_units", 0) > 0:
            return ["HARVEST"]
    return None

def unit_act(pos, mine, me, seeds, day, held):
    fx, fy = pos
    here = me["tiles"][fy][fx]
    if (fx, fy) in mine and here is not None:
        act = tile_needs(here, seeds, day)
        if act: return act
    for (x, y) in mine:
        t = me["tiles"][y][x]
        if (x, y) != (fx, fy) and t is not None and tile_needs(t, seeds, day):
            return step_toward((fx, fy), (x, y))
    if held.get(CARROT, 0) >= 9:
        return ["DROP"] if (fx, fy) == SHED else step_toward((fx, fy), SHED)
    if (fx, fy) in mine and here is None and seeds.get(CARROT, 0) > 0:
        return ["PLANT", CARROT]
    for (x, y) in mine:
        if (x, y) != (fx, fy) and me["tiles"][y][x] is None and seeds.get(CARROT, 0) > 0:
            return step_toward((fx, fy), (x, y))
    if held.get(CARROT, 0) > 0:
        return ["DROP"] if (fx, fy) == SHED else step_toward((fx, fy), SHED)
    return ["PASS"]

def agent(obs):
    me = obs["farms"][obs["player"]]
    private = obs.get("private", {})
    seeds = private.get("seeds", {})
    invs = private.get("inventories") or [{}]
    hands = me["hands"]
    units = 1 + len(hands)

    market = []
    shed_stock = private.get("shed", {}).get(CARROT, 0)
    if shed_stock > 0: market.append(["SELL", CARROT, shed_stock])
    empty = sum(1 for (x, y) in PATCH if me["tiles"][y][x] is None)
    need = empty - seeds.get(CARROT, 0)
    if need > 0 and me["money"] >= 20 * need:
        market.append(["BUY_SEED", CARROT, need])
    if BUY_LAND and "NE" not in me["unlocked_quadrants"] and me["money"] >= 1400:
        market.append(["BUY_LAND"])
    # The crew goes home at nightfall, so a bot that wants hands re-hires every morning. One
    # order per hand. This bot asks for one a turn, so it reaches its crew over several turns;
    # six HIRE entries in one turn would hire six at once, at the Fibonacci price of the sixth.
    if len(hands) < CREW and me["money"] >= 300:
        market.append(["HIRE"])

    shares = [PATCH[i::units] for i in range(units)]      # nobody walks to another unit's plant
    acts = [unit_act(me["farmer"], shares[0], me, seeds, obs["day"], invs[0])]
    for i, h in enumerate(hands):
        pos = h["pos"] if isinstance(h, dict) else h
        acts.append(unit_act(pos, shares[i + 1], me, seeds, obs["day"],
                             invs[i + 1] if len(invs) > i + 1 else {}))
    return {"farmer": acts[0], "hands": acts[1:], "market": market}
