"""Kestrel: a template-A style Kaggriculture agent with an explicit task scheduler.

Day 0 replays the opening shared by the top ladder teams (cows + sheep on pastures by the shed, 5 hands,
melons + wheat, cash spent to ~0). From day 1 a scheduler builds every tile task for the day
(feed / care / collect fertilizer / harvest animals, fertilize-then-water plants, harvest crops at peak,
dig weeds, build structures, place animals, plant just-in-time) and assigns them greedily to the farmer
and hired hands by travel distance, handling shed pickups (wheat, fertilizer, animals) and drops.
Selling is aligned to the town-shop consumption ticks (step % 4 == 1) in small, price-guarded lots.
"""
import math

# ----------------------------------------------------------------------------- engine constants
CROPS = {
    "WHEAT": {"seed": 10, "first": 2, "maxd": 4, "interval": 0, "maxy": 6, "ongoing": False},
    "CARROT": {"seed": 20, "first": 2, "maxd": 3, "interval": 0, "maxy": 4, "ongoing": False},
    "TOMATO": {"seed": 50, "first": 8, "maxd": 8, "interval": 1, "maxy": 4, "ongoing": True},
    "STRAWBERRY": {"seed": 100, "first": 10, "maxd": 10, "interval": 2, "maxy": 4, "ongoing": True},
    "MELON": {"seed": 80, "first": 10, "maxd": 12, "interval": 0, "maxy": 6, "ongoing": False},
}
ANIMALS = {
    "GOOSE": {"cost": 300, "st": "COOP", "first": 4, "interval": 1, "held": 4, "prod": "EGG"},
    "COW": {"cost": 400, "st": "PASTURE", "first": 8, "interval": 2, "held": 6, "prod": "MILK"},
    "SHEEP": {"cost": 500, "st": "PASTURE", "first": 6, "interval": 3, "held": 6, "prod": "WOOL"},
}
MARKET = {
    "WHEAT": (25, 400, "sqrt", 0.80, "log", 0.20), "CARROT": (35, 450, "hinge", 1.00, "sqrt", 0.70),
    "TOMATO": (60, 200, "hinge", 0.40, "sqrt", 0.60), "STRAWBERRY": (120, 100, "sqrt", 0.70, "linear", 1.60),
    "MELON": (250, 300, "log", 0.20, "sq", 3.60), "EGG": (50, 332, "hinge", 0.40, "log", 0.20),
    "MILK": (160, 122, "sqrt", 0.60, "linear", 1.60), "WOOL": (200, 105, "log", 0.20, "sq", 3.20),
    "FERTILIZER": (100, 200, "linear", 0.40, "linear", 0.40),
}
I0 = 10000
SHOPS = {
    "BAKERY": ["EGG", "WHEAT"], "PIZZA_SHOP": ["MILK", "TOMATO", "WHEAT"],
    "BRUNCH_SPOT": ["EGG", "WHEAT", "STRAWBERRY"], "YARN_STORE": ["WOOL"],
    "ICE_CREAM_SHOP": ["STRAWBERRY", "MILK", "WHEAT"], "PET_CAFE": ["CARROT"],
    "SMOOTHIE_SHOP": ["STRAWBERRY", "MILK"], "FARMERS_MARKET": ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY"],
}
MOVES = {"NORTH": (0, -1), "SOUTH": (0, 1), "EAST": (1, 0), "WEST": (-1, 0)}
SHED_TILES = [(4, 4), (5, 4), (4, 5), (5, 5)]
LAND_ORDER = ["NE", "SW", "SE"]
LAND_PRICES = [1000, 2000, 4000]
LAST_STEP = 718
PREMIUM = ("STRAWBERRY", "MILK", "WOOL", "MELON")


def _shape(f, x, T):
    x = max(0.0, x)
    if f == "linear":
        return x
    if f == "sq":
        return x * x
    if f == "sqrt":
        return math.sqrt(x)
    if f == "log":
        return math.log(1.0 + x)
    if f == "hinge":
        u = x / T
        return u + 8.0 * max(0.0, u - 1.0) ** 2
    return x


def price_at(item, inv):
    base, T, bf, bt, af, at = MARKET[item]
    if inv < I0:
        p = base + bt * base / _shape(bf, T, T) * _shape(bf, I0 - inv, T)
    else:
        p = base - at * base / _shape(af, T, T) * _shape(af, inv - I0, T)
    return max(1, int(round(p)))


def quadrant(x, y):
    return ("N" if y < 5 else "S") + ("W" if x < 5 else "E")


def fib(n):
    a, b = 1, 1
    for _ in range(n):
        a, b = b, a + b
    return a


def dist(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def step_toward(p, q):
    if p[0] < q[0]:
        return "EAST"
    if p[0] > q[0]:
        return "WEST"
    if p[1] < q[1]:
        return "SOUTH"
    if p[1] > q[1]:
        return "NORTH"
    return "PASS"


def nearest_shed(p):
    return min(SHED_TILES, key=lambda s: (dist(p, s), SHED_TILES.index(s)))


# ----------------------------------------------------------------------------- day-0 opening tape
# The top template's day 0 (identical across DSM / Vadim games): 2 cows + 3 sheep on 5 pastures by the
# shed, 5 hands, 6 melons + 12 wheat planted and watered, cash spent.
OPENING = [
    {"farmer": ["PASS"], "hands": [], "market": [["BUY_ANIMAL", "COW", 1], ["BUY_PRODUCT", "WHEAT", 5], ["BUY_ANIMAL", "SHEEP", 1], ["BUY_ANIMAL", "SHEEP", 1], ["BUY_ANIMAL", "SHEEP", 1]]},
    {"farmer": ["PICKUP", "COW", 1], "hands": [], "market": [["SELL", "WHEAT", 1], ["HIRE"], ["HIRE"], ["HIRE"], ["HIRE"], ["BUY_ANIMAL", "COW", 1], ["HIRE"]]},
    {"farmer": ["BUILD_PASTURE"], "hands": [["PICKUP", "SHEEP", 1], ["PICKUP", "SHEEP", 1], ["PICKUP", "COW", 1], ["PICKUP", "SHEEP", 1], ["PICKUP", "COW", 1]], "market": [["SELL", "WHEAT", 1]]},
    {"farmer": ["PLACE", "COW", 1], "hands": [["NORTH"], ["NORTH"], ["NORTH"], ["WEST"], ["NORTH"]], "market": [["SELL", "WHEAT", 1], ["BUY_PRODUCT", "WHEAT", 1]]},
    {"farmer": ["PICKUP", "WHEAT", 3], "hands": [["WEST"], ["WEST"], ["NORTH"], ["BUILD_PASTURE"], ["NORTH"]], "market": [["BUY_PRODUCT", "WHEAT", 1]]},
    {"farmer": ["CARE"], "hands": [["BUILD_PASTURE"], ["PLACE", "SHEEP", 1], ["NORTH"], ["PLACE", "SHEEP", 1], ["NORTH"]], "market": [["BUY_SEED", "MELON", 2], ["BUY_PRODUCT", "WHEAT", 1]]},
    {"farmer": ["WEST"], "hands": [["PLACE", "SHEEP", 1], ["CARE"], ["WEST"], ["WEST"], ["WEST"]], "market": [["BUY_SEED", "MELON", 2]]},
    {"farmer": ["FEED"], "hands": [["CARE"], ["WEST"], ["BUILD_PASTURE"], ["BUILD_PASTURE"], ["PLANT", "MELON"]], "market": []},
    {"farmer": ["EAST"], "hands": [["SOUTH"], ["NORTH"], ["PLACE", "COW", 1], ["PLACE", "SHEEP", 1], ["WATER"]], "market": [["BUY_SEED", "WHEAT", 1]]},
    {"farmer": ["NORTH"], "hands": [["NORTH"], ["PLANT", "MELON"], ["NORTH"], ["WEST"], ["NORTH"]], "market": [["BUY_SEED", "WHEAT", 1]]},
    {"farmer": ["FEED"], "hands": [["WEST"], ["WATER"], ["WEST"], ["PLANT", "MELON"], ["PLANT", "WHEAT"]], "market": [["BUY_SEED", "MELON", 2]]},
    {"farmer": ["SOUTH"], "hands": [["PLANT", "MELON"], ["NORTH"], ["PLANT", "WHEAT"], ["WATER"], ["WATER"]], "market": [["BUY_SEED", "WHEAT", 3]]},
    {"farmer": ["WEST"], "hands": [["WATER"], ["PLANT", "WHEAT"], ["WATER"], ["WEST"], ["WEST"]], "market": [["BUY_SEED", "MELON", 2]]},
    {"farmer": ["WEST"], "hands": [["NORTH"], ["WATER"], ["WEST"], ["PLANT", "MELON"], ["PLANT", "WHEAT"]], "market": [["BUY_SEED", "WHEAT", 1]]},
    {"farmer": ["FEED"], "hands": [["PLANT", "MELON"], ["WEST"], ["PLANT", "WHEAT"], ["WATER"], ["WATER"]], "market": [["SELL", "WHEAT", 2]]},
    {"farmer": ["CARE"], "hands": [["WATER"], ["PASS"], ["WATER"], ["NORTH"], ["WEST"]], "market": [["BUY_SEED", "WHEAT", 3]]},
    {"farmer": ["PASS"], "hands": [["WEST"], ["PLANT", "WHEAT"], ["WEST"], ["NORTH"], ["PLANT", "WHEAT"]], "market": [["BUY_SEED", "WHEAT", 1]]},
    {"farmer": ["PASS"], "hands": [["WEST"], ["WATER"], ["PLANT", "WHEAT"], ["PLANT", "WHEAT"], ["WATER"]], "market": [["BUY_SEED", "WHEAT", 1]]},
    {"farmer": ["PASS"], "hands": [["WEST"], ["WEST"], ["WATER"], ["WATER"], ["WEST"]], "market": [["BUY_SEED", "WHEAT", 1]]},
    {"farmer": ["PASS"], "hands": [["NORTH"], ["NORTH"], ["WEST"], ["NORTH"], ["PLANT", "WHEAT"]], "market": [["BUY_SEED", "WHEAT", 1]]},
    {"farmer": ["PASS"], "hands": [["PLANT", "WHEAT"], ["NORTH"], ["NORTH"], ["NORTH"], ["WATER"]], "market": [["BUY_SEED", "WHEAT", 1]]},
    {"farmer": ["PASS"], "hands": [["WATER"], ["PLANT", "WHEAT"], ["PASS"], ["PASS"], ["PASS"]], "market": [["BUY_SEED", "WHEAT", 1]]},
    {"farmer": ["PASS"], "hands": [["SOUTH"], ["WATER"], ["PASS"], ["PASS"], ["SOUTH"]], "market": []},
    {"farmer": ["PASS"], "hands": [["PASS"], ["PASS"], ["PASS"], ["PASS"], ["PASS"]], "market": []},
    {"farmer": ["COLLECT_FERTILIZER"], "hands": [], "market": [["HIRE"], ["HIRE"], ["HIRE"]]},
    {"farmer": ["PLACE", "FERTILIZER", 1], "hands": [["NORTH"], ["NORTH"], ["NORTH"]], "market": []},
    {"farmer": ["NORTH"], "hands": [["NORTH"], ["NORTH"], ["NORTH"]], "market": [["SELL", "FERTILIZER", 1], ["BUY_PRODUCT", "WHEAT", 3]]},
    {"farmer": ["COLLECT_FERTILIZER"], "hands": [["WEST"], ["WEST"], ["WEST"]], "market": []},
    {"farmer": ["SOUTH"], "hands": [["COLLECT_FERTILIZER"], ["SOUTH"], ["WEST"]], "market": []},
    {"farmer": ["DROP"], "hands": [["SOUTH"], ["COLLECT_FERTILIZER"], ["SOUTH"]], "market": []},
    {"farmer": ["PICKUP", "WHEAT", 2], "hands": [["SOUTH"], ["EAST"], ["WEST"]], "market": [["SELL", "FERTILIZER", 1], ["BUY_PRODUCT", "WHEAT", 2]]},
    {"farmer": ["FEED"], "hands": [["DROP"], ["DROP"], ["COLLECT_FERTILIZER"]], "market": [["BUY_SEED", "MELON", 1]]},
    {"farmer": ["CARE"], "hands": [["PICKUP", "WHEAT", 2], ["PICKUP", "WHEAT", 2], ["EAST"]], "market": [["SELL", "FERTILIZER", 2], ["BUY_PRODUCT", "WHEAT", 2]]},
    {"farmer": ["NORTH"], "hands": [["NORTH"], ["NORTH"], ["CARE"]], "market": [["BUY_SEED", "MELON", 1]]},
    {"farmer": ["CARE"], "hands": [["CARE"], ["CARE"], ["EAST"]], "market": [["BUY_SEED", "MELON", 1]]},
    {"farmer": ["FEED"], "hands": [["NORTH"], ["WEST"], ["DROP"]], "market": []},
    {"farmer": ["WEST"], "hands": [["CARE"], ["WEST"], ["PICKUP", "WHEAT", 2]], "market": [["SELL", "FERTILIZER", 1]]},
    {"farmer": ["WEST"], "hands": [["FEED"], ["SOUTH"], ["WEST"]], "market": []},
    {"farmer": ["WEST"], "hands": [["SOUTH"], ["CARE"], ["FEED"]], "market": []},
    {"farmer": ["PLANT", "MELON"], "hands": [["PASS"], ["FEED"], ["PASS"]], "market": []},
    {"farmer": ["WATER"], "hands": [["PASS"], ["WEST"], ["PASS"]], "market": []},
    {"farmer": ["WEST"], "hands": [["PASS"], ["WEST"], ["PASS"]], "market": []},
    {"farmer": ["PLANT", "MELON"], "hands": [["PASS"], ["PASS"], ["PASS"]], "market": []},
    {"farmer": ["WATER"], "hands": [["PASS"], ["PASS"], ["PASS"]], "market": []},
    {"farmer": ["PASS"], "hands": [["PASS"], ["PASS"], ["PASS"]], "market": []},
    {"farmer": ["PASS"], "hands": [["PASS"], ["PASS"], ["PASS"]], "market": []},
    {"farmer": ["PASS"], "hands": [["PASS"], ["PASS"], ["PASS"]], "market": []},
    {"farmer": ["PASS"], "hands": [["PASS"], ["PASS"], ["PASS"]], "market": []},
    {"farmer": ["COLLECT_FERTILIZER"], "hands": [], "market": [["SELL", "WHEAT", 2], ["HIRE"], ["HIRE"], ["HIRE"], ["HIRE"], ["HIRE"]]},
    {"farmer": ["PLACE", "FERTILIZER", 1], "hands": [["NORTH"], ["NORTH"], ["NORTH"], ["NORTH"], ["NORTH"]], "market": []},
    {"farmer": ["WEST"], "hands": [["WEST"], ["WEST"], ["NORTH"], ["COLLECT_FERTILIZER"], ["NORTH"]], "market": [["SELL", "FERTILIZER", 1]]},
    {"farmer": ["COLLECT_FERTILIZER"], "hands": [["NORTH"], ["WEST"], ["WEST"], ["SOUTH"], ["WEST"]], "market": []},
    {"farmer": ["EAST"], "hands": [["COLLECT_FERTILIZER"], ["COLLECT_FERTILIZER"], ["CARE"], ["DROP"], ["NORTH"]], "market": []},
    {"farmer": ["DROP"], "hands": [["SOUTH"], ["EAST"], ["NORTH"], ["NORTH"], ["NORTH"]], "market": [["SELL", "FERTILIZER", 2]]},
    {"farmer": ["WEST"], "hands": [["SOUTH"], ["EAST"], ["NORTH"], ["NORTH"], ["WATER"]], "market": []},
    {"farmer": ["WEST"], "hands": [["DROP"], ["DROP"], ["WATER"], ["WEST"], ["WEST"]], "market": []},
    {"farmer": ["WEST"], "hands": [["CARE"], ["WEST"], ["WEST"], ["WATER"], ["WATER"]], "market": [["SELL", "FERTILIZER", 2], ["BUY_ANIMAL", "COW", 1]]},
    {"farmer": ["WATER"], "hands": [["PICKUP", "COW", 1], ["CARE"], ["WATER"], ["WEST"], ["WEST"]], "market": []},
    {"farmer": ["WEST"], "hands": [["WEST"], ["NORTH"], ["WEST"], ["WATER"], ["WATER"]], "market": []},
    {"farmer": ["WATER"], "hands": [["NORTH"], ["WATER"], ["WATER"], ["WEST"], ["WEST"]], "market": []},
    {"farmer": ["NORTH"], "hands": [["NORTH"], ["WEST"], ["HARVEST"], ["WATER"], ["WATER"]], "market": []},
    {"farmer": ["NORTH"], "hands": [["NORTH"], ["WATER"], ["SOUTH"], ["NORTH"], ["HARVEST"]], "market": [["BUY_SEED", "STRAWBERRY", 1]]},
    {"farmer": ["WATER"], "hands": [["HARVEST"], ["PASS"], ["SOUTH"], ["WATER"], ["PLANT", "STRAWBERRY"]], "market": [["BUY_SEED", "STRAWBERRY", 1]]},
    {"farmer": ["NORTH"], "hands": [["BUILD_PASTURE"], ["PASS"], ["SOUTH"], ["WEST"], ["WATER"]], "market": []},
    {"farmer": ["WATER"], "hands": [["PLACE", "COW", 1], ["PASS"], ["CARE"], ["NORTH"], ["EAST"]], "market": []},
    {"farmer": ["HARVEST"], "hands": [["SOUTH"], ["PASS"], ["FEED"], ["WATER"], ["SOUTH"]], "market": []},
    {"farmer": ["PLANT", "STRAWBERRY"], "hands": [["SOUTH"], ["PASS"], ["EAST"], ["HARVEST"], ["EAST"]], "market": [["BUY_SEED", "WHEAT", 1]]},
    {"farmer": ["WATER"], "hands": [["EAST"], ["NORTH"], ["FEED"], ["PASS"], ["SOUTH"]], "market": []},
    {"farmer": ["EAST"], "hands": [["FEED"], ["NORTH"], ["PASS"], ["PASS"], ["SOUTH"]], "market": [["BUY_SEED", "WHEAT", 1]]},
    {"farmer": ["PASS"], "hands": [["PASS"], ["PLANT", "WHEAT"], ["PASS"], ["PLANT", "WHEAT"], ["SOUTH"]], "market": []},
    {"farmer": ["SOUTH"], "hands": [["SOUTH"], ["WATER"], ["PASS"], ["WATER"], ["PASS"]], "market": []},
    {"farmer": ["PASS"], "hands": [["PASS"], ["PASS"], ["PASS"], ["PASS"], ["PASS"]], "market": []},
    {"farmer": ["COLLECT_FERTILIZER"], "hands": [], "market": [["SELL", "WHEAT", 7], ["HIRE"], ["HIRE"], ["HIRE"], ["HIRE"], ["HIRE"]]},
    {"farmer": ["DROP"], "hands": [["NORTH"], ["NORTH"], ["NORTH"], ["NORTH"], ["NORTH"]], "market": []},
    {"farmer": ["WEST"], "hands": [["WEST"], ["WEST"], ["NORTH"], ["COLLECT_FERTILIZER"], ["NORTH"]], "market": [["SELL", "FERTILIZER", 1]]},
    {"farmer": ["COLLECT_FERTILIZER"], "hands": [["NORTH"], ["WEST"], ["NORTH"], ["SOUTH"], ["WEST"]], "market": []},
    {"farmer": ["EAST"], "hands": [["COLLECT_FERTILIZER"], ["COLLECT_FERTILIZER"], ["WEST"], ["DROP"], ["NORTH"]], "market": []},
    {"farmer": ["DROP"], "hands": [["SOUTH"], ["EAST"], ["NORTH"], ["NORTH"], ["NORTH"]], "market": [["SELL", "FERTILIZER", 1]]},
    {"farmer": ["WEST"], "hands": [["SOUTH"], ["EAST"], ["WEST"], ["NORTH"], ["WATER"]], "market": [["SELL", "FERTILIZER", 1]]},
    {"farmer": ["WEST"], "hands": [["DROP"], ["DROP"], ["COLLECT_FERTILIZER"], ["NORTH"], ["HARVEST"]], "market": []},
    {"farmer": ["NORTH"], "hands": [["CARE"], ["NORTH"], ["SOUTH"], ["WEST"], ["SOUTH"]], "market": [["SELL", "FERTILIZER", 2], ["BUY_ANIMAL", "COW", 1]]},
    {"farmer": ["NORTH"], "hands": [["PICKUP", "COW", 1], ["CARE"], ["SOUTH"], ["NORTH"], ["SOUTH"]], "market": []},
    {"farmer": ["WATER"], "hands": [["NORTH"], ["WEST"], ["EAST"], ["WATER"], ["FEED"]], "market": []},
    {"farmer": ["WEST"], "hands": [["NORTH"], ["WEST"], ["SOUTH"], ["HARVEST"], ["CARE"]], "market": []},
    {"farmer": ["WATER"], "hands": [["NORTH"], ["WEST"], ["DROP"], ["SOUTH"], ["SOUTH"]], "market": [["BUY_SEED", "STRAWBERRY", 1]]},
    {"farmer": ["NORTH"], "hands": [["NORTH"], ["WATER"], ["PASS"], ["FEED"], ["FEED"]], "market": [["SELL", "FERTILIZER", 1], ["BUY_SEED", "STRAWBERRY", 1]]},
    {"farmer": ["WATER"], "hands": [["BUILD_PASTURE"], ["WEST"], ["PASS"], ["CARE"], ["SOUTH"]], "market": [["BUY_SEED", "STRAWBERRY", 1]]},
    {"farmer": ["HARVEST"], "hands": [["PLACE", "COW", 1], ["WATER"], ["PASS"], ["SOUTH"], ["FEED"]], "market": []},
    {"farmer": ["PLANT", "STRAWBERRY"], "hands": [["WEST"], ["NORTH"], ["PASS"], ["SOUTH"], ["PASS"]], "market": [["BUY_SEED", "STRAWBERRY", 1]]},
    {"farmer": ["WATER"], "hands": [["PLANT", "STRAWBERRY"], ["WATER"], ["PASS"], ["SOUTH"], ["PASS"]], "market": []},
    {"farmer": ["SOUTH"], "hands": [["WATER"], ["HARVEST"], ["PASS"], ["CARE"], ["PASS"]], "market": []},
    {"farmer": ["HARVEST"], "hands": [["WEST"], ["PLANT", "STRAWBERRY"], ["PASS"], ["FEED"], ["PASS"]], "market": []},
    {"farmer": ["PLANT", "STRAWBERRY"], "hands": [["WATER"], ["WATER"], ["PASS"], ["WEST"], ["PASS"]], "market": []},
    {"farmer": ["WATER"], "hands": [["PASS"], ["PASS"], ["PASS"], ["FEED"], ["PASS"]], "market": []},
    {"farmer": ["PASS"], "hands": [["WEST"], ["PASS"], ["PASS"], ["CARE"], ["PASS"]], "market": []},
    {"farmer": ["PASS"], "hands": [["WATER"], ["PASS"], ["PASS"], ["PASS"], ["PASS"]], "market": []},
]
N_OPEN = len(OPENING)

# ----------------------------------------------------------------------------- plan parameters
P = {
    # animal targets by day: list of (day_from, count)
    "COW": [(0, 2), (2, 3), (3, 4), (4, 5), (5, 6), (7, 7)],
    "SHEEP": [(0, 3), (3, 4), (6, 5), (8, 6)],
    "GOOSE": [(6, 3), (8, 7)],
    "last_animal_day": 17,
    "strawberry_total": 32, "strawberry_first_day": 2, "strawberry_last_day": 13,
    "melon_total": 12, "melon_last_day": 6,
    "carrot_first_day": 10, "carrot_last_day": 26, "carrot_share": 0.35,
    "wheat_last_day": 25,
    "land_ne_day": 5, "land_sw_day": 7, "land_se": False,
    "reserve": 60,               # cash kept aside for feed/seeds
    "sheep_retire_day": 21,
    "max_hands": 12,
}

# pasture / coop slots nearest the shed first (NW), then NE column 5, then SW for coops.
PASTURE_SLOTS = [(4, 4), (3, 4), (4, 3), (3, 3), (2, 4), (2, 3), (4, 2), (3, 2), (2, 2), (4, 1),
                 (5, 4), (5, 3), (5, 2), (6, 4), (6, 3), (1, 3), (1, 4)]
COOP_SLOTS = [(4, 5), (3, 5), (4, 6), (2, 5), (3, 6), (4, 7), (2, 6), (1, 5), (5, 1), (6, 2)]


def target_count(kind, day):
    n = 0
    for d, c in P[kind]:
        if day >= d:
            n = c
    return n


# ----------------------------------------------------------------------------- persistent state
class Mem:
    def __init__(self):
        self.last_step = -1
        self.planted = {c: 0 for c in CROPS}      # seeds planted so far (by crop)
        self.bought_animals = {a: 0 for a in ANIMALS}
        self.plan_crop = {}                        # (x,y) -> crop reserved for an empty tile today
        self.sold_log = []
        self.exec_prev = {}
        self.commit = {}
        self.commit_day = -1
        self.zone_n = -1
        self.zones = {}
        self.keepers = {0}


M = Mem()


def g(o, k, d=None):
    if isinstance(o, dict):
        return o.get(k, d)
    return getattr(o, k, d)


# ----------------------------------------------------------------------------- helpers on state
def tile_kind(t):
    if t is None:
        return "EMPTY"
    if t == "LOCKED":
        return "LOCKED"
    return t.get("kind")


def animals_on_farm(tiles):
    cnt = {a: 0 for a in ANIMALS}
    for row in tiles:
        for t in row:
            if isinstance(t, dict) and t.get("animal"):
                cnt[t["animal"]] += 1
    return cnt


def shop_demand(shops):
    d = {p: 1.0 / 24 for p in MARKET if p != "FERTILIZER"}   # per turn
    for s in shops:
        prods = SHOPS.get(s, [])
        m = 2 if len(prods) == 1 else 1
        for p in prods:
            d[p] = d.get(p, 0) + m / 4.0
    return d


class Brain:
    def __init__(self, obs, step):
        self.obs = obs
        self.step = step
        self.day = step // 24
        self.hour = step % 24
        self.me = g(obs, "player")
        self.farm = g(obs, "farms")[self.me]
        self.priv = g(obs, "private")
        self.tiles = self.farm["tiles"]
        self.money = float(self.farm["money"])
        self.shed = dict(self.priv["shed"])
        self.seeds = dict(self.priv["seeds"])
        self.units = [tuple(self.farm["farmer"])] + [tuple(h) for h in self.farm["hands"]]
        self.invs = [dict(i) for i in self.priv["inventories"]]
        while len(self.invs) < len(self.units):
            self.invs.append({})
        self.market_inv = dict(g(obs, "market")["inventory"])
        self.prices = dict(g(obs, "market")["prices"])
        self.shops = list(g(obs, "town")["unlocked_shops"])
        self.unlocked = list(self.farm["unlocked_quadrants"])
        self.hires_today = int(self.farm.get("hires_today", 0))
        self.orders = []
        self.cash = self.money
        self.sim_inv = dict(self.market_inv)   # market inventory after our planned sells
        self.unfed = 0

    # ---------------------------------------------------------------- scans
    def scan(self):
        self.animal_tiles, self.plant_tiles, self.empty, self.weeds = [], [], [], []
        self.structs_free = {"PASTURE": [], "COOP": []}
        for y in range(10):
            for x in range(10):
                t = self.tiles[y][x]
                k = tile_kind(t)
                if k == "EMPTY":
                    self.empty.append((x, y))
                elif k == "WEED":
                    self.weeds.append((x, y))
                elif k == "PLANT":
                    self.plant_tiles.append((x, y))
                elif k in ("PASTURE", "COOP"):
                    (self.animal_tiles if t.get("animal") else self.structs_free[k]).append((x, y))
        self.animals = animals_on_farm(self.tiles)
        self.carried = {}
        for inv in self.invs:
            for k, v in inv.items():
                self.carried[k] = self.carried.get(k, 0) + v

    def age(self, t):
        return self.day - t["planted_day"]

    def have(self, item):
        return self.shed.get(item, 0) + self.carried.get(item, 0)

    # ---------------------------------------------------------------- tasks
    def build_tasks(self):
        T = {}
        day, hour = self.day, self.hour
        fert_have = self.have("FERTILIZER")
        self.unfed = 0
        for (x, y) in self.animal_tiles:
            t = self.tiles[y][x]
            a = t["animal"]
            info = ANIMALS[a]
            retire = a == "SHEEP" and day >= P["sheep_retire_day"] and self.prices.get("WOOL", 0) < 60 \
                and t.get("consecutive_unfed", 0) == 0 and t.get("yield_units", 0) == 0
            ops = []
            if not t["fed_today"] and not retire and day < 29:
                ops.append(("FEED", None, "WHEAT", 9 + (30 if t["consecutive_unfed"] >= 1 else 0)))
                self.unfed += 1
            if not t["cared_today"] and not retire and day < 29:
                ops.append(("CARE", None, None, 5))
            if t.get("fertilizer_available"):
                ops.append(("COLLECT_FERTILIZER", None, None, 4))
            yu = t.get("yield_units", 0)
            per = 1 + info["interval"]
            if yu > 0 and (yu >= info["held"] - per + 1 or day >= 28 or (yu >= 2 and hour >= 16)):
                ops.append(("HARVEST", None, None, 6))
            if ops:
                T[(x, y)] = ops
        for (x, y) in self.plant_tiles:
            t = self.tiles[y][x]
            c = t["crop"]
            cd = CROPS[c]
            age = self.age(t)
            yu = t.get("yield_units", 0)
            decaying = t["max_lifespan_step"] >= 0 and self.step >= t["max_lifespan_step"]
            ops = []
            want_fert = (t.get("fertilized_until_day", -1) < day and not t["watered_today"] and (
                (c in ("WHEAT", "CARROT") and age == 2) or (c == "STRAWBERRY" and age in (9, 13)) or
                (c == "TOMATO" and age in (7, 10)) or (c == "MELON" and age == 6)))
            if want_fert and fert_have > 0:
                ops.append(("FERTILIZE", None, "FERTILIZER", 8))
                fert_have -= 1
            must = t.get("consecutive_unwatered", 0) >= 1      # skipping today would turn it into a weed
            if not cd["ongoing"]:
                ready = age >= cd["first"] and (yu >= cd["maxy"] or decaying or
                                               (age >= cd["maxd"] and t["watered_today"]) or
                                               (c == "MELON" and age >= 10 and t["watered_today"]) or
                                               (self.step >= LAST_STEP - 30 and t["watered_today"]))
                in_window = (cd["maxd"] + 1) // 2 <= age <= cd["maxd"] and yu < cd["maxy"]
                if ready:
                    ops.append(("HARVEST", None, None, 10 if decaying else 8))
                elif not t["watered_today"] and (must or in_window):
                    ops.append(("WATER", None, None, 11 if must else 8))
            else:
                dsf = day + 1 - t["planted_day"] - cd["first"]
                prod_tonight = dsf >= 0 and dsf % cd["interval"] == 0 and dsf // cd["interval"] + 1 <= cd["maxy"]
                fert_now = t.get("fertilized_until_day", -1) >= day
                if not t["watered_today"] and not decaying and (must or (prod_tonight and fert_now)):
                    ops.append(("WATER", None, None, 11 if must else 8))
                if yu > 0 and (yu >= 2 or decaying or day >= 28 or yu >= cd["maxy"] - 1):
                    ops.append(("HARVEST", None, None, 9 if decaying else 6))
            if ops:
                T[(x, y)] = ops
        return T

    # ---------------------------------------------------------------- crop choice
    def choose_crop(self, counts):
        day = self.day
        # a crop must be harvestable before the end
        s_tot = M.planted["STRAWBERRY"] + counts.get("STRAWBERRY", 0)
        if P["strawberry_first_day"] <= day <= P["strawberry_last_day"] and s_tot < P["strawberry_total"]:
            return "STRAWBERRY"
        m_tot = M.planted["MELON"] + counts.get("MELON", 0)
        if 1 <= day <= P["melon_last_day"] and m_tot < P["melon_total"]:
            return "MELON"
        if P["carrot_first_day"] <= day <= P["carrot_last_day"]:
            if counts["CARROT_active"] < P["carrot_share"] * counts["crop_tiles"]:
                counts["CARROT_active"] += 1
                return "CARROT"
        if day <= P["wheat_last_day"]:
            return "WHEAT"
        if day <= P["carrot_last_day"]:
            return "CARROT"
        return None

    # ---------------------------------------------------------------- main
    def run(self):
        self.scan()
        day, hour, step = self.day, self.hour, self.step
        final = step >= LAST_STEP - 5
        tasks = {} if final else self.build_tasks()

        # --- structures and animals
        want = {}
        if day <= P["last_animal_day"] and not final:
            for a in ANIMALS:
                want[a] = max(0, target_count(a, day) - self.animals[a] - self.have(a))
        need_st = {"PASTURE": 0, "COOP": 0}
        for a in ANIMALS:
            need_st[ANIMALS[a]["st"]] += self.have(a) + want.get(a, 0)
        reserved = set()
        for kind, slots in (("PASTURE", PASTURE_SLOTS), ("COOP", COOP_SLOTS)):
            extra = need_st[kind] - len(self.structs_free[kind])
            for s in slots:
                if extra <= 0:
                    break
                if self.tiles[s[1]][s[0]] is None:
                    tasks.setdefault(s, []).append(("BUILD_" + kind, None, None, 7))
                    reserved.add(s)
                    extra -= 1
        for kind in ("PASTURE", "COOP"):
            free = list(self.structs_free[kind])
            for a in ANIMALS:
                if ANIMALS[a]["st"] != kind:
                    continue
                for _ in range(self.have(a)):
                    if not free:
                        break
                    s = free.pop(0)
                    tasks.setdefault(s, []).append(("PLACE", a, a, 10))

        # --- planting jobs on empty tiles
        counts = {"crop_tiles": len(self.plant_tiles) + len(self.empty),
                  "CARROT_active": sum(1 for (x, y) in self.plant_tiles if self.tiles[y][x]["crop"] == "CARROT")}
        plant_jobs = []
        if not final and hour <= 20:
            empties = sorted((e for e in self.empty if e not in reserved), key=lambda p: dist(p, (4, 4)))
            for e in empties:
                c = self.choose_crop(counts)
                if c is None:
                    break
                counts[c] = counts.get(c, 0) + 1
                plant_jobs.append((e, c))
        if not final and day <= 27:
            for w in self.weeds:
                tasks.setdefault(w, []).append(("DIG", None, None, 3))

        # --- market (order matters: sells fund the buys that follow)
        self.plan_sells(final)
        self.plan_feed()
        if not final:
            self.plan_hires(tasks, plant_jobs)
        plant_now = self.plan_seeds(plant_jobs)
        for (e, c) in plant_now:
            tasks.setdefault(e, []).insert(0, ("PLANT", c, "SEED:" + c, 8))
        if not final:
            self.plan_land_and_animals(want)

        actions = self.assign(tasks, final)
        for a in actions:
            if a and a[0] == "PLANT":
                M.planted[a[1]] += 1
        return {"farmer": actions[0], "hands": actions[1:], "market": self.orders[:10]}

    def reserve_cash(self):
        return P["reserve"] + 30 * max(0, self.unfed - self.have("WHEAT"))

    # ---------------------------------------------------------------- selling
    def sell(self, item, n):
        n = int(min(n, self.shed.get(item, 0)))
        if n <= 0:
            return 0
        rev = 0
        inv = self.sim_inv[item]
        for k in range(n):
            p = price_at(item, inv)
            rev += p
            if p > 1:
                inv += 1
        self.sim_inv[item] = inv
        self.orders.append(["SELL", item, n])
        self.shed[item] -= n
        self.cash += rev * 0.95
        return n

    def plan_sells(self, final):
        tick = self.step % 4 == 1
        for item in ("STRAWBERRY", "MILK", "WOOL", "MELON", "TOMATO", "EGG", "CARROT", "WHEAT", "FERTILIZER"):
            have = self.shed.get(item, 0)
            if item == "WHEAT":
                have -= self.wheat_keep()
            elif item == "FERTILIZER":
                have -= self.fert_keep()
            if have <= 0:
                continue
            n = have if final else self.sell_qty(item, have, tick)
            if n > 0:
                self.sell(item, n)
        # liquidity: if cash cannot cover feed + a minimal crew, sell fertilizer / wheat surplus now
        need = 30 * max(0, self.unfed - self.have("WHEAT")) + 25
        for item in ("FERTILIZER", "EGG", "WHEAT", "CARROT"):
            if self.cash >= need:
                break
            spare = self.shed.get(item, 0) - (self.wheat_keep() if item == "WHEAT" else 0)
            k = 0
            while spare - k > 0 and self.cash + k * self.prices.get(item, 1) * 0.9 < need:
                k += 1
            if k:
                self.sell(item, k)

    def sell_qty(self, item, have, tick):
        base = MARKET[item][0]
        days_left = (LAST_STEP - self.step) / 24.0
        demand = shop_demand(self.shops).get(item, 0) * 24.0      # units/day the town absorbs
        if item in PREMIUM:
            if not tick and days_left > 1.0:
                return 0
            frac = {"STRAWBERRY": 0.95, "MILK": 0.5, "WOOL": 0.5, "MELON": 0.75}[item]
            lot = 3 if item == "MELON" else 4
        elif item == "FERTILIZER":
            frac, lot = 0.3, 3
        elif item == "EGG":
            frac, lot = 0.6, 6
        else:
            frac, lot = 0.7, 8
        # stock the town cannot absorb before the end will never fetch a better price: sell it
        absorb = demand * max(0.0, days_left - 0.5)
        if have > absorb + 6 or item == "FERTILIZER":
            frac *= 0.3
            lot += int(min(12, have - absorb))
        if days_left < 1.0:
            frac *= 0.25
            lot = max(lot, int(math.ceil(have / max(1.0, days_left * 6))))
        elif days_left < 3:
            frac *= 0.7
            lot += 2
        if sum(self.shed.values()) > 70:
            frac *= 0.5
            lot += 8
        thr = max(2, frac * base)
        inv = self.sim_inv[item]
        n = 0
        while n < min(have, lot):
            if price_at(item, inv + n) < thr:
                break
            n += 1
        return n

    def wheat_keep(self):
        if self.day >= 29:
            return 0
        return min(len(self.animal_tiles) + 3, 40)

    def fert_keep(self):
        if self.day >= 28:
            return 0
        need = 0
        for (x, y) in self.plant_tiles:
            t = self.tiles[y][x]
            age = self.age(t) + 1
            c = t["crop"]
            if (c in ("WHEAT", "CARROT") and age == 2) or (c == "STRAWBERRY" and age in (9, 13)) or \
                    (c == "MELON" and age == 6) or (c == "TOMATO" and age in (7, 10)):
                need += 1
        return need + 2

    # ---------------------------------------------------------------- buying
    def plan_feed(self):
        wheat = self.have("WHEAT")
        n_anim = len(self.animal_tiles)
        need = self.unfed + (n_anim if self.hour >= 20 and self.day < 29 else 0)
        short = need - wheat
        if short > 0:
            p = self.prices.get("WHEAT", 25) + 2
            n = int(min(short, max(0, (self.cash - 5) // p)))
            if n > 0:
                self.orders.append(["BUY_PRODUCT", "WHEAT", n])
                self.cash -= n * p

    def workload(self, tasks, plant_jobs):
        w = 0.0
        for pos, ops in tasks.items():
            w += len(ops) + 1.6
        w += 3.5 * len(plant_jobs)
        w += 0.35 * self.unfed + 1.0
        return w

    def plan_hires(self, tasks, plant_jobs):
        turns_left = 23 - self.hour
        if turns_left < 4:
            return
        work = self.workload(tasks, plant_jobs)
        want_units = int(math.ceil(work / (turns_left * 0.75)))
        want_units = max(1, min(P["max_hands"] + 1, want_units))
        n_now = len(self.units)
        h = self.hires_today
        while n_now < want_units:
            c = fib(h)
            if self.cash - c < 0:
                break
            self.orders.append(["HIRE"])
            self.cash -= c
            h += 1
            n_now += 1
        self.n_units_next = n_now

    def plan_seeds(self, plant_jobs):
        seeds_left = dict(self.seeds)
        plant_now = []
        for (e, c) in plant_jobs:
            if seeds_left.get(c, 0) > 0:
                seeds_left[c] -= 1
                plant_now.append((e, c))
        cap = max(0, getattr(self, "n_units_next", len(self.units)) - len(plant_now))
        buy = {}
        for (e, c) in plant_jobs[len(plant_now):]:
            if cap <= 0:
                break
            cost = CROPS[c]["seed"]
            land = self.land_due() or 0
            if self.cash - cost < self.reserve_cash() + (0.5 * land if c in ("STRAWBERRY", "MELON") else 0):
                break
            self.cash -= cost
            buy[c] = buy.get(c, 0) + 1
            cap -= 1
        for c, n in buy.items():
            self.orders.append(["BUY_SEED", c, n])
        return plant_now

    def land_due(self):
        n_extra = len(self.unlocked) - 1
        if n_extra >= 3 or self.day > 14:
            return None
        q = LAND_ORDER[n_extra]
        ok_day = {"NE": P["land_ne_day"], "SW": P["land_sw_day"], "SE": 10 if P["land_se"] else 99}[q]
        if self.day < ok_day:
            return None
        return LAND_PRICES[n_extra]

    def struct_capacity(self, kind):
        """free structures + buildable slots in unlocked land"""
        n = len(self.structs_free[kind])
        slots = PASTURE_SLOTS if kind == "PASTURE" else COOP_SLOTS
        for (x, y) in slots:
            t = self.tiles[y][x]
            if t is None:
                n += 1
        return n

    def plan_land_and_animals(self, want):
        cost = self.land_due()
        if cost is not None:
            if self.cash - cost >= self.reserve_cash():
                self.orders.append(["BUY_LAND"])
                self.cash -= cost
            else:
                return      # save up for land before buying animals
        cap = {"PASTURE": self.struct_capacity("PASTURE"), "COOP": self.struct_capacity("COOP")}
        for a in ANIMALS:
            cap[ANIMALS[a]["st"]] -= self.have(a)
        for a in ("COW", "SHEEP", "GOOSE"):
            st = ANIMALS[a]["st"]
            for _ in range(want.get(a, 0)):
                c = ANIMALS[a]["cost"]
                if cap[st] <= 0 or self.cash - c < self.reserve_cash() + 30:
                    break
                self.orders.append(["BUY_ANIMAL", a, 1])
                self.cash -= c
                cap[st] -= 1

    # ---------------------------------------------------------------- assignment
    def make_zones(self, n, tasks):
        """Animal tiles form dedicated keeper zones (about 8 animals each); crop tiles are split
        into contiguous snake-order strips of balanced workload for the remaining units."""
        order = []
        for y in range(10):
            xs = range(10) if y % 2 == 0 else range(9, -1, -1)
            for x in xs:
                if self.tiles[y][x] != "LOCKED":
                    order.append((x, y))
        animal = [t for t in order if tile_kind(self.tiles[t[1]][t[0]]) in ("PASTURE", "COOP")]
        crops = [t for t in order if t not in set(animal)]
        zones = {}
        self_keepers = set()
        if n <= 1:
            for t in order:
                zones[t] = 0
            M.keepers = {0}
            return zones
        k_a = min(n - 1, max(1, int(math.ceil(len(animal) / 8.0)))) if animal else 0
        # keeper zones: split animal tiles by count
        for j, t in enumerate(animal):
            zones[t] = min(k_a - 1, j * k_a // max(1, len(animal)))
        M.keepers = set(range(k_a))
        rest = n - k_a
        w = []
        for t in crops:
            ops = tasks.get(t)
            if ops:
                w.append(1.0 + len(ops))
            else:
                k = tile_kind(self.tiles[t[1]][t[0]])
                w.append(1.6 if k == "PLANT" else 0.3)
        total = sum(w)
        target = total / max(1, rest)
        acc, zi = 0.0, 0
        for t, wt in zip(crops, w):
            if acc >= target * (zi + 1) and zi < rest - 1:
                zi += 1
            zones[t] = k_a + zi
            acc += wt
        return zones

    def assign(self, tasks, final):
        n = len(self.units)
        acts = [["PASS"] for _ in range(n)]
        taken = set()
        shed_left = dict(self.shed)
        seeds_used = {}
        if M.commit_day != self.day or M.zone_n != n:
            M.commit = {}
            M.commit_day = self.day
            M.zone_n = n
            # the farmer works alone at hour 0; zones are for the full crew
            M.zones = self.make_zones(n, tasks)
        zones = M.zones
        # per-zone item needs
        need_by_zone = {}
        for tpos, ops in tasks.items():
            z = zones.get(tpos, 0)
            d = need_by_zone.setdefault(z, {"WHEAT": 0, "FERTILIZER": 0})
            for op in ops:
                if op[2] in ("WHEAT", "FERTILIZER"):
                    d[op[2]] += 1
        for i in range(n):
            pos = self.units[i]
            inv = self.invs[i]
            if final:
                if any(v > 0 for v in inv.values()):
                    acts[i] = ["DROP"] if pos in SHED_TILES else [step_toward(pos, nearest_shed(pos))]
                continue
            produce = sum(v for k, v in inv.items() if k not in ("WHEAT", "FERTILIZER") and k not in ANIMALS)
            fert_c = inv.get("FERTILIZER", 0)
            if pos in SHED_TILES and sum(self.shed.values()) < 97 and not any(inv.get(a, 0) for a in ANIMALS) and \
                    (produce > 0 or fert_c >= 6 or (fert_c and self.money < 80)):
                acts[i] = ["DROP"]
                continue
            my_zone = i
            best = None
            prev = M.commit.get(i)
            for tpos, ops in tasks.items():
                if tpos in taken:
                    continue
                op = None
                detour = 0
                for cand in ops:
                    need = cand[2]
                    if need == "WHEAT" and i not in M.keepers and inv.get("WHEAT", 0) <= 0 and self.hour < 20:
                        continue
                    if need in ("WHEAT", "FERTILIZER") or need in ANIMALS:
                        if inv.get(need, 0) <= 0:
                            if shed_left.get(need, 0) <= 0:
                                continue
                            sh = nearest_shed(pos)
                            detour = dist(pos, sh) + 1 + dist(sh, tpos) - dist(pos, tpos)
                    elif need and need.startswith("SEED:"):
                        c = need[5:]
                        if self.seeds.get(c, 0) - seeds_used.get(c, 0) <= 0:
                            continue
                    op = cand
                    break
                if op is None:
                    continue
                d = dist(pos, tpos) + detour
                prio = max(o[3] for o in ops if o[2] is None or o is op)
                in_zone = zones.get(tpos, -1) == my_zone
                urgent = self.hour >= 18 and any(o[0] in ("WATER", "FEED") for o in ops)
                score = -d + 0.25 * prio + (8 if in_zone else 0) + (6 if urgent else 0) + (2 if prev == tpos else 0)
                if best is None or score > best[0]:
                    best = (score, tpos, op, detour)
            if best is None:
                if produce and pos not in SHED_TILES:
                    acts[i] = [step_toward(pos, nearest_shed(pos))]
                continue
            _, tpos, op, detour = best
            taken.add(tpos)
            M.commit[i] = tpos
            need = op[2]
            if detour > 0:
                if pos in SHED_TILES:
                    zneed = need_by_zone.get(zones.get(tpos, my_zone), {}).get(need, 1)
                    if need == "WHEAT":
                        k = max(1, min(shed_left.get("WHEAT", 0), max(2, zneed), 12))
                    elif need == "FERTILIZER":
                        k = max(1, min(shed_left.get("FERTILIZER", 0), max(1, zneed), 6))
                    else:
                        k = 1
                    shed_left[need] = shed_left.get(need, 0) - k
                    acts[i] = ["PICKUP", need, int(k)]
                else:
                    acts[i] = [step_toward(pos, nearest_shed(pos))]
                continue
            if pos != tpos:
                acts[i] = [step_toward(pos, tpos)]
                continue
            name, arg = op[0], op[1]
            if name == "PLANT":
                seeds_used[arg] = seeds_used.get(arg, 0) + 1
                acts[i] = ["PLANT", arg]
            elif name == "PLACE":
                acts[i] = ["PLACE", arg, 1]
            else:
                acts[i] = [name]
        return acts


def _act(obs, step):
    me = g(obs, "player")
    farm = g(obs, "farms")[me]
    if step < N_OPEN:
        a = OPENING[step]
        for u in [a.get("farmer")] + list(a.get("hands") or []):
            if u and u[0] == "PLANT":
                M.planted[u[1]] += 1
        hands = [list(h) for h in a["hands"]]
        n_h = len(g(farm, "hands", []))
        hands = (hands + [["PASS"]] * n_h)[:n_h]
        return {"farmer": list(a["farmer"]), "hands": hands, "market": [list(m) for m in a["market"]]}
    return Brain(obs, step).run()


# ----------------------------------------------------------------------------- the agent
def agent(obs, config=None):
    step = g(obs, "step", None)
    if step is None:
        step = g(obs, "day", 0) * 24 + g(obs, "hour", 0)
    global M
    if step == 0 or step < M.last_step:
        M = Mem()
    M.last_step = step
    try:
        return _act(obs, step)
    except Exception:
        return {"farmer": ["PASS"], "hands": [["PASS"]] * len(g(g(obs, "farms")[g(obs, "player")], "hands", [])),
                "market": []}
