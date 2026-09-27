from collections import defaultdict

CROPS = {
    "WHEAT":      {"seed": 10,  "first": 2,  "maxage": 4,  "base": 25},
    "CARROT":     {"seed": 20,  "first": 2,  "maxage": 3,  "base": 35},
    "TOMATO":     {"seed": 50,  "first": 8,  "maxage": 11, "base": 60},
    "STRAWBERRY": {"seed": 100, "first": 10, "maxage": 16, "base": 120},
    "MELON":      {"seed": 80,  "first": 10, "maxage": 10, "base": 250},
}
ANIMALS = {
    "GOOSE": {"cost": 300, "struct": "COOP",    "first": 4},
    "COW":   {"cost": 400, "struct": "PASTURE", "first": 8},
    "SHEEP": {"cost": 500, "struct": "PASTURE", "first": 6},
}
LAND = [1000, 2000, 4000]
LAST_DAY = 29
TUNE = {'cap':40,'reserve':1200,'wheat_px':80,'hands':14,'melon':10,'flock':24,'landday':6,'seedstock':30}

STATE = {"day": -1, "assign": {}}


def mv(p, t):
    dx, dy = t[0] - p[0], t[1] - p[1]
    if dx == 0 and dy == 0:
        return "PASS"
    if abs(dx) >= abs(dy):
        return "EAST" if dx > 0 else "WEST"
    return "SOUTH" if dy > 0 else "NORTH"


def dist(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def fib(n):
    a, b = 1, 1
    for _ in range(max(0, n)):
        a, b = b, a + b
    return a


class Plan:
    def __init__(self, obs):
        self.obs = obs
        self.me = obs["player"]
        self.day = obs["day"]
        self.hour = obs["hour"]
        self.farm = obs["farms"][self.me]
        self.tiles = self.farm["tiles"]
        self.n = len(self.tiles)
        self.money = float(self.farm.get("money", 0))
        pv = obs.get("private", {}) or {}
        self.shed = dict(pv.get("shed") or {})
        self.seeds = dict(pv.get("seeds") or {})
        self.invs = pv.get("inventories") or []
        mk = obs.get("market", {}) or {}
        self.prices = mk.get("prices") or {}
        self.minv = mk.get("inventory") or {}

        self.units = [tuple(self.farm.get("farmer", [0, 0]))]
        self.units += [tuple(h) for h in (self.farm.get("hands") or [])]

        h = self.n // 2
        self.shed_tiles = [(h - 1, h - 1), (h, h - 1), (h - 1, h), (h, h)]

        self.empty, self.weeds, self.plants = [], [], []
        self.animals, self.free_struct = [], []
        self.unlocked = 0
        for y, row in enumerate(self.tiles):
            for x, t in enumerate(row):
                p = (x, y)
                if t == "LOCKED":
                    continue
                self.unlocked += 1
                if t is None:
                    self.empty.append(p)
                elif isinstance(t, dict):
                    k = t.get("kind")
                    if k == "WEED":
                        self.weeds.append(p)
                    elif k == "PLANT":
                        self.plants.append((p, t))
                    elif k in ("COOP", "PASTURE"):
                        if t.get("animal"):
                            self.animals.append((p, t))
                        else:
                            self.free_struct.append((p, k))

        self.ncrop = defaultdict(int)
        for _, t in self.plants:
            self.ncrop[t.get("crop")] += 1
        self.nanimal = defaultdict(int)
        for _, t in self.animals:
            self.nanimal[t.get("animal")] += 1

    def price(self, item):
        try:
            return float(self.prices.get(item, 0) or 0)
        except Exception:
            return 0.0

    def inv(self, i):
        if i < len(self.invs) and isinstance(self.invs[i], dict):
            return self.invs[i]
        return {}

    def shed_spot(self, pos):
        return min(self.shed_tiles, key=lambda p: dist(pos, p))

    def want_coop(self):
        nan = len(self.animals) + sum(int(self.shed.get(a, 0)) for a in ANIMALS)
        return (self.day <= 24 and nan + len(self.free_struct) < TUNE["flock"]
                and len(self.free_struct) < 2)

    def near_shed_tile(self, pos):
        return min(dist(pos, s) for s in self.shed_tiles) <= 3

    def best_build_spot(self, pos):
        return min(self.empty, key=lambda p: (
            min(dist(p, s) for s in self.shed_tiles) * 2 + dist(pos, p)))

    def age(self, t):
        return self.day - int(t.get("planted_day", self.day))

    def ready(self, t):
        """Harvest only at full value, not at first yield."""
        crop = t.get("crop")
        if crop not in CROPS or int(t.get("yield_units", 0)) <= 0:
            return False
        a = self.age(t)
        c = CROPS[crop]
        if crop in ("TOMATO", "STRAWBERRY"):
            return True
        if self.day >= LAST_DAY - 1:
            return True
        return a >= c["maxage"]


# ---------------- jobs ----------------

def build_jobs(P):
    """(value, kind, target, need_item) - value is coins-ish per action."""
    jobs = []
    end = P.day >= LAST_DAY - 1

    for pos, t in P.animals:
        a = t.get("animal")
        if int(t.get("yield_units", 0)) > 0:
            prod = {"GOOSE": "EGG", "COW": "MILK", "SHEEP": "WOOL"}[a]
            jobs.append((max(20.0, P.price(prod)) * min(2, int(t["yield_units"])),
                         "HARVEST", pos, None))
        if not t.get("fed_today") and not end:
            urgent = int(t.get("consecutive_unfed", 0)) >= 1
            jobs.append((900.0 if urgent else 400.0, "FEED", pos, "WHEAT"))
        if t.get("fertilizer_available"):
            jobs.append((max(10.0, P.price("FERTILIZER")), "COLLECT_FERTILIZER", pos, None))
        if not t.get("cared_today") and not end:
            jobs.append((36.0, "CARE", pos, None))

    for pos, t in P.plants:
        crop = t.get("crop")
        if P.ready(t):
            jobs.append((max(10.0, P.price(crop)) * int(t.get("yield_units", 1)),
                         "HARVEST", pos, None))
        elif not t.get("watered_today") and not end:
            v = 55.0 if crop == "MELON" else 22.0
            if int(t.get("consecutive_unwatered", 0)) >= 1:
                v += 40.0
            jobs.append((v, "WATER", pos, None))

    # place animals waiting in shed onto free structures
    for a in ANIMALS:
        if int(P.shed.get(a, 0)) > 0:
            for pos, k in P.free_struct:
                if k == ANIMALS[a]["struct"]:
                    jobs.append((150.0, "PLACE", pos, a))

    if not end:
        for pos in P.weeds:
            jobs.append((14.0, "DIG", pos, None))
    return jobs


def crop_wanted(P):
    """What to plant on a fresh empty tile."""
    if P.day > 25:
        return None
    if P.day + 10 <= 27 and P.price("MELON") > 110 and P.ncrop["MELON"] < TUNE["melon"] \
            and int(P.seeds.get("MELON", 0)) > 0:
        return "MELON"
    # wheat is feed first, cash second: grow enough to cover the flock
    if int(P.seeds.get("WHEAT", 0)) > 0 and P.day + 4 <= 28:
        return "WHEAT"
    return None


def assign(P):
    """Greedy assignment, but sticky: a unit keeps its target until done.

    Recomputing from scratch every turn made units thrash between targets as
    job values shifted, burning ~80% of all actions on movement."""
    jobs = sorted(build_jobs(P), key=lambda j: -j[0])
    live = {(k, t): (v, n) for v, k, t, n in jobs}
    out, taken = {}, set()
    prev = STATE["assign"]

    for i in range(len(P.units)):
        key = prev.get(i)
        if key and key in live and key not in taken:
            need = live[key][1]
            if need and int(P.inv(i).get(need, 0)) <= 0:
                continue
            out[i] = (key[0], key[1], need)
            taken.add(key)

    free = set(range(len(P.units))) - set(out)
    for val, kind, tgt, need in jobs:
        if not free:
            break
        if (kind, tgt) in taken:
            continue
        best, bd = None, 1e9
        for i in free:
            if need and int(P.inv(i).get(need, 0)) <= 0:
                continue
            d = dist(P.units[i], tgt)
            if d < bd:
                best, bd = i, d
        if best is None or val - 3.0 * bd <= 0:
            continue
        out[best] = (kind, tgt, need)
        taken.add((kind, tgt))
        free.discard(best)

    STATE["assign"] = {i: (k, t) for i, (k, t, _) in out.items()}
    return out, free


def logistics(P, i, pos):
    """Unassigned unit: deploy animals, haul, plant. Order matters."""
    inv = P.inv(i)
    tile = P.tiles[pos[1]][pos[0]]
    at_shed = pos in P.shed_tiles
    carried = sum(int(v or 0) for v in inv.values())
    holding = [a for a in ANIMALS if int(inv.get(a, 0)) > 0]

    # 1. Carrying livestock beats everything. Never DROP while holding an
    #    animal - that dumps it back in the shed and loops forever.
    if holding:
        a = holding[0]
        need = ANIMALS[a]["struct"]
        if isinstance(tile, dict) and tile.get("kind") == need and not tile.get("animal"):
            return ["PLACE", a, 1]
        spots = [p for p, k in P.free_struct if k == need]
        if spots:
            return [mv(pos, min(spots, key=lambda p: dist(pos, p)))]
        if tile is None:
            return ["BUILD_" + need]
        if P.empty:
            return [mv(pos, P.best_build_spot(pos))]
        return ["PASS"]

    # 2. dump a full load
    if carried >= 10 or (P.day >= LAST_DAY - 1 and carried > 0):
        if at_shed:
            return ["DROP"]
        return [mv(pos, P.shed_spot(pos))]

    # 3. collect one waiting animal (one at a time, so a drop can't strand a flock)
    if P.empty or P.free_struct:
        for a in ANIMALS:
            if int(P.shed.get(a, 0)) > 0:
                if at_shed:
                    return ["PICKUP", a, 1]
                return [mv(pos, P.shed_spot(pos))]

    # 4. stock wheat for the feeding crew
    unfed = sum(1 for _, t in P.animals if not t.get("fed_today"))
    carried_wheat = sum(int(P.inv(j).get("WHEAT", 0)) for j in range(len(P.units)))
    if unfed and carried_wheat < unfed * 2 and int(inv.get("WHEAT", 0)) == 0 \
            and int(P.shed.get("WHEAT", 0)) > 0 and P.day < LAST_DAY - 1:
        if at_shed:
            return ["PICKUP", "WHEAT", min(20, int(P.shed.get("WHEAT", 0)))]
        return [mv(pos, P.shed_spot(pos))]

    # 5. keep a coop standing so a bought goose deploys the same turn
    if tile is None and P.want_coop():
        return ["BUILD_COOP"]

    crop = crop_wanted(P)
    if crop:
        if tile is None and P.claim(crop):
            return ["PLANT", crop]
        if P.empty:
            tgt = min(P.empty, key=lambda p: dist(pos, p))
            if tgt != pos:
                return [mv(pos, tgt)]
    if carried > 0:
        if at_shed:
            return ["DROP"]
        return [mv(pos, P.shed_spot(pos))]
    return ["PASS"]


# ---------------- market ----------------

def market(P):
    orders, money = [], P.money
    end = P.day >= LAST_DAY - 1
    nanimals = sum(P.nanimal.values()) + sum(int(P.shed.get(a, 0)) for a in ANIMALS)

    # hire at the top of the day - labour is the bottleneck and fib is cheap
    if P.hour == 0:
        want = 4 if P.day < 2 else min(TUNE["hands"], 3 + P.unlocked // 6 + nanimals)
        k = int(P.farm.get("hires_today", 0))
        have = len(P.units)
        while have < want and len(orders) < 9:
            c = fib(k)
            if money < c + 400:
                break
            orders.append(["HIRE"])
            money -= c
            have += 1
            k += 1

    # sell
    sell = []
    feed_reserve = 0 if end else nanimals * 3 + 6
    for item in ("EGG", "FERTILIZER", "MELON", "MILK", "WOOL", "STRAWBERRY",
                 "TOMATO", "CARROT", "WHEAT"):
        q = int(P.shed.get(item, 0) or 0)
        if item == "WHEAT":
            q -= feed_reserve
        if q <= 0:
            continue
        p = P.price(item)
        if not end:
            base = CROPS.get(item, {}).get("base", 50)
            if item in ("MELON", "STRAWBERRY", "MILK", "WOOL") and p < 0.45 * base:
                q = min(q, 1)
        sell.append((p * q, ["SELL", item, q]))
    sell.sort(reverse=True, key=lambda s: s[0])
    orders += [o for _, o in sell[:5]]

    if end:
        return orders[:10]

    # seeds
    if len(orders) < 9:
        if P.price("MELON") > 90 and P.day + 10 <= 27 and int(P.seeds.get("MELON", 0)) < 6 \
                and money > 1200:
            orders.append(["BUY_SEED", "MELON", 6])
            money -= 480
        want_w = 16 if P.day < 26 else 0
        if int(P.seeds.get("WHEAT", 0)) < want_w and money > 150:
            q = min(want_w, int((money - 100) // 10))
            if q > 0:
                orders.append(["BUY_SEED", "WHEAT", q])
                money -= 10 * q

    # feed wheat - only top up when it is genuinely cheap. Buying at $50 to
    # feed birds whose eggs sell at $46 is a negative-margin trade.
    if len(orders) < 10 and nanimals:
        have = int(P.shed.get("WHEAT", 0))
        need = nanimals * 4 + 14 - have
        wp = max(1.0, P.price("WHEAT"))
        if need > 0 and money > 500 and wp <= TUNE["wheat_px"]:
            q = min(need, int((money - 400) // wp), 40)
            if q > 0:
                orders.append(["BUY_PRODUCT", "WHEAT", q])
                money -= q * wp

    # geese: the engine. buy as many as cash and tiles allow.
    if len(orders) < 10 and P.day + 5 <= 27:
        room = len(P.empty) + len(P.free_struct)
        backlog = sum(int(P.shed.get(a, 0)) for a in ANIMALS)
        carried_a = sum(int(P.inv(j).get(a, 0)) for j in range(len(P.units)) for a in ANIMALS)
        if (nanimals < TUNE["flock"] and len(P.free_struct) >= 1
                and money >= TUNE["reserve"] and backlog == 0 and carried_a == 0):
            q = min(len(P.free_struct), 2, int((money - TUNE["reserve"]) // 300) + 1)
            if q > 0:
                orders.append(["BUY_ANIMAL", "GOOSE", q])
                money -= 300 * q

    # land
    if len(orders) < 10:
        owned = len(P.farm.get("unlocked_quadrants") or [])
        idx = owned - 1
        if idx == 0 and P.day >= TUNE["landday"] and money >= 1000 + TUNE["reserve"]:
            orders.append(["BUY_LAND"])
            money -= 1000
    return orders[:10]


def agent(obs):
    P = Plan(obs)
    P._planted = defaultdict(int)

    def claim(crop):
        if P._planted[crop] < int(P.seeds.get(crop, 0)):
            P._planted[crop] += 1
            return True
        return False
    P.claim = claim

    jobs, free = assign(P)
    acts = []
    for i, pos in enumerate(P.units):
        if i in jobs:
            kind, tgt, need = jobs[i]
            if pos == tgt:
                acts.append([kind, need, 1] if kind == "PLACE" else [kind])
            else:
                acts.append([mv(pos, tgt)])
        else:
            acts.append(logistics(P, i, pos))
    return {"farmer": acts[0] if acts else ["PASS"],
            "hands": acts[1:],
            "market": market(P)}


__all__ = ["agent"]