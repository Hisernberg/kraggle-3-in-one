"""FlyFarmer — Drosophila MaleCNS v1.0 connectome が農場を経営する Kaggriculture エージェント。

    観測 → [人間設計] 刺激エンコーダ → 視覚投射ニューロン LPLC2 / LC4 (somaSide L/R)
        → [connectome] 728 ニューロン・35,704 シナプス、synapse count 重み × 神経伝達物質符号、学習なし
        → 下行ニューロン DNp01 (Giant Fiber) ... の左右活動 → [人間設計] デコーダ → 経営判断
        → [反射弓] 移動・水やり・収穫などの手足

connectome 部分（重み・接続）は一切手で書いていない。人間が設計したのは「何を刺激とみなすか」と
「下行ニューロンの活動を何の判断に写すか」だけ。
"""
from __future__ import annotations
import os
import numpy as np

import sys

def _find_weights(name="brain_weights.npz"):
    """kaggle_environments はファイルを exec するので __file__ が無い。sys.path (提出物 dir が末尾に追加される)・
    Kaggle の agent dir・cwd を順に探す。"""
    cands = [os.environ.get("FLY_WEIGHTS", "")]
    cands += [os.path.join(p, name) for p in reversed(sys.path) if p]
    cands += [os.path.join("/kaggle_simulations/agent", name), os.path.join(os.getcwd(), name)]
    try:
        cands.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), name))
    except NameError:
        pass
    for c in cands:
        if c and os.path.exists(c):
            return c
    raise FileNotFoundError(name)

try:
    WEIGHTS = _find_weights()   # ローダは exec 後に sys.path を戻すので、import 時に解決しておく
except FileNotFoundError:
    WEIGHTS = None

CROPS = {  # kaggle_environments の CROPS と同値
    "WHEAT":      {"seed": 10,  "first": 2,  "max": 4,  "ongoing": False},
    "CARROT":     {"seed": 20,  "first": 2,  "max": 3,  "ongoing": False},
    "TOMATO":     {"seed": 50,  "first": 8,  "max": 11, "ongoing": True},
    "STRAWBERRY": {"seed": 100, "first": 10, "max": 16, "ongoing": True},
    "MELON":      {"seed": 80,  "first": 10, "max": 10, "ongoing": False},
}
BASE_PRICE = {"WHEAT": 25, "CARROT": 35, "TOMATO": 60, "STRAWBERRY": 120, "MELON": 250,
              "EGG": 50, "MILK": 160, "WOOL": 200, "FERTILIZER": 100}
SHED_TILE = (4, 4)
# NW 区画で shed に近い順の 8 マス（農夫 1 人で 1 日 24 手に収まる規模）
MANAGED = [(4, 4), (3, 4), (4, 3), (3, 3), (2, 4), (4, 2), (2, 3), (3, 2)]
LAST_DAY = 29


# ---------------------------------------------------------------- connectome
class FlyBrain:
    """rate model: r <- (1-decay) r + decay * clip(W r + I, 0, 1)。状態は試合を通じて持続する。"""

    def __init__(self, path=None, decay=0.2, threshold=0.05, steps_per_turn=3,
                 shuffle_seed=None, silence=()):
        z = np.load(path or WEIGHTS or _find_weights(), allow_pickle=False)
        self.W = z["W"].astype(np.float32)
        self.role, self.side, self.type = z["role"], z["side"], z["type"]
        if shuffle_seed is not None:            # 対照実験: 接続を保ったまま重みだけ並べ替える
            rng = np.random.default_rng(shuffle_seed)
            nz = np.flatnonzero(self.W)
            self.W.flat[nz] = rng.permutation(self.W.flat[nz])
        self.silence = np.isin(self.type, list(silence))
        self.decay, self.threshold, self.steps = decay, threshold, steps_per_turn
        sens = self.role == "sensory"; desc = self.role == "descending"
        self.sens_L, self.sens_R = sens & (self.side == "L"), sens & (self.side == "R")
        self.desc_L, self.desc_R = desc & (self.side == "L"), desc & (self.side == "R")
        self.desc_idx = np.flatnonzero(desc)
        self.r = np.zeros(len(self.W), dtype=np.float32)

    def step(self, left: float, right: float, looming: float) -> np.ndarray:
        I = np.zeros_like(self.r)
        I[self.sens_L] = min(1.0, left + looming)
        I[self.sens_R] = min(1.0, right + looming)
        for _ in range(self.steps):
            out = np.where(self.silence, 0.0, self.r)
            target = np.clip(self.W @ out + I, 0.0, 1.0)
            target[target < self.threshold] = 0.0
            self.r = (1 - self.decay) * self.r + self.decay * target
            self.r[self.silence] = 0.0
        return self.r

    def readout(self) -> dict:
        L = float(self.r[self.desc_L].mean()); R = float(self.r[self.desc_R].mean())
        yaw = float(np.tanh((R - L) / max(L + R, 0.3)))     # -1 = 左 (畑側), +1 = 右 (市場側)
        pitch = float(np.tanh(0.8 * (L + R) / 2))           # 両側同時活動 = looming 逃避
        gf = float(self.r[(self.type == "DNp01")].mean())   # Giant Fiber
        return {"L": L, "R": R, "yaw": yaw, "pitch": pitch, "gf": gf,
                "sens_L": float(self.r[self.sens_L].mean()), "sens_R": float(self.r[self.sens_R].mean())}


# ---------------------------------------------------------------- 人間設計: 刺激
def encode_stimulus(obs, farm, private) -> tuple[float, float, float]:
    """左視野 = 畑の用事、右視野 = 市場の売り時、looming = 脅威（雑草・相手の資金差）。"""
    tiles = farm.get("tiles", [])
    chores = 0
    weeds = 0
    for (x, y) in MANAGED:
        t = tiles[y][x] if y < len(tiles) and x < len(tiles[y]) else None
        if isinstance(t, dict):
            if t.get("kind") == "WEED":
                weeds += 1
            elif t.get("kind") == "PLANT":
                if not t.get("watered_today") or t.get("yield_units", 0) > 0:
                    chores += 1
    left = min(1.0, chores / 6.0)

    shed = private.get("shed", {}) or {}
    prices = (obs.get("market", {}) or {}).get("prices", {}) or {}
    stock = [(k, v) for k, v in shed.items() if v > 0 and k in BASE_PRICE]
    if stock:
        ratio = sum(v * prices.get(k, BASE_PRICE[k]) / BASE_PRICE[k] for k, v in stock) / sum(v for _, v in stock)
        right = min(1.0, ratio * min(1.0, sum(v for _, v in stock) / 8.0))
    else:
        right = 0.0

    farms = obs.get("farms", [])
    me = obs.get("player", 0)
    opp_money = farms[1 - me].get("money", 0) if len(farms) > 1 else farm.get("money", 0)
    gap = max(0.0, (opp_money - farm.get("money", 0)) / 4000.0)
    looming = min(1.0, weeds / 2.0 + gap)
    return left, right, looming


# ---------------------------------------------------------------- 人間設計: デコーダ
def decide(ro: dict, day: int) -> dict:
    """DN 活動 → 経営判断。"""
    escape = ro["pitch"] > 0.5 or ro["gf"] > 0.6           # 逃避反射: 何でも売って雑草を掘る
    sell = escape or ro["yaw"] > 0.05 or day >= LAST_DAY    # 右 DN 優位 → 市場へ
    arousal = (ro["L"] + ro["R"]) / 2
    # 興奮した蠅は足の速い作物、落ち着いた蠅は長期作物を選ぶ
    if arousal > 0.35:
        crop = "WHEAT"
    elif arousal > 0.15:
        crop = "CARROT"
    elif arousal > 0.05:
        crop = "TOMATO"
    else:
        crop = "STRAWBERRY"
    return {"sell": sell, "escape": escape, "crop": crop, "arousal": arousal}


# ---------------------------------------------------------------- 反射弓: 手足
def _move_toward(pos, target):
    x, y = pos; tx, ty = target
    if x < tx: return ["EAST"]
    if x > tx: return ["WEST"]
    if y < ty: return ["SOUTH"]
    if y > ty: return ["NORTH"]
    return None

def _ripe(tile, day):
    c = CROPS.get(tile.get("crop"))
    if c is None: return False
    age = day - tile.get("planted_day", day)
    if tile.get("yield_units", 0) <= 0: return False
    if c["ongoing"]: return True
    return age >= c["max"] or day >= LAST_DAY

def _plantable(crop, day):
    c = CROPS[crop]
    return day + c["first"] <= LAST_DAY - 1

def body(obs, farm, private, decision) -> tuple[list, list]:
    day, hour = obs.get("day", 0), obs.get("hour", 0)
    tiles = farm.get("tiles", [])
    pos = tuple(farm.get("farmer", SHED_TILE))
    inv = (private.get("inventories") or [{}])[0] or {}
    carried = sum(v for k, v in inv.items() if k != "seeds")
    seeds = private.get("seeds", {}) or {}
    shed = private.get("shed", {}) or {}
    money = farm.get("money", 0)

    def tile_at(p):
        x, y = p
        return tiles[y][x] if y < len(tiles) and x < len(tiles[y]) else "LOCKED"

    # 目標マスを優先度付きで集める
    weeds, ripe, thirsty, empty = [], [], [], []
    for p in MANAGED:
        t = tile_at(p)
        if t == "LOCKED": continue
        if t is None:
            empty.append(p)
        elif t.get("kind") == "WEED":
            weeds.append(p)
        elif t.get("kind") == "PLANT":
            if _ripe(t, day): ripe.append(p)
            elif not t.get("watered_today", False): thirsty.append(p)

    # 手持ちが溜まった / 用事が無い → shed に戻して DROP
    if carried > 0 and (carried >= 4 or not (ripe or thirsty) or hour >= 22):
        if pos == SHED_TILE:
            farmer = ["DROP"]
        else:
            farmer = _move_toward(pos, SHED_TILE)
    else:
        order = (weeds + ripe + thirsty + empty) if decision["escape"] else (ripe + thirsty + weeds + empty)
        farmer = ["PASS"]
        for p in order:
            t = tile_at(p)
            if t is None and not (seeds.get(decision["crop"], 0) > 0 and hour < 23 and _plantable(decision["crop"], day)):
                continue
            mv = _move_toward(pos, p)
            if mv:
                farmer = mv; break
            if t is None:
                farmer = ["PLANT", decision["crop"]]
            elif t.get("kind") == "WEED":
                farmer = ["DIG"]
            elif _ripe(t, day):
                farmer = ["HARVEST"]
            else:
                farmer = ["WATER"]
            break

    # 市場
    market = []
    if decision["sell"]:
        for k, v in shed.items():
            if v > 0 and k in BASE_PRICE:
                market.append(["SELL", k, int(v)])
    crop = decision["crop"]
    if empty and seeds.get(crop, 0) == 0 and money >= CROPS[crop]["seed"] and _plantable(crop, day):
        market.append(["BUY_SEED", crop, 1])
    return farmer, market[:10]


# ---------------------------------------------------------------- entry point
_BRAIN = None
TRACE = []   # notebook 用: 1 試合ぶんの脳活動ログ

def make_agent(brain: FlyBrain | None = None, trace: list | None = None):
    def agent(obs, config=None):
        farms = obs.get("farms", [])
        player = obs.get("player", 0)
        if not farms or player >= len(farms):
            return {"farmer": ["PASS"], "hands": [], "market": []}
        farm = farms[player]
        private = obs.get("private", {}) or {}
        b = brain if brain is not None else _get_brain()
        left, right, looming = encode_stimulus(obs, farm, private)
        b.step(left, right, looming)
        ro = b.readout()
        decision = decide(ro, obs.get("day", 0))
        farmer, market = body(obs, farm, private, decision)
        if trace is not None:
            trace.append({"step": obs.get("step", 0), "left": left, "right": right, "looming": looming,
                          **ro, "sell": decision["sell"], "escape": decision["escape"],
                          "crop": decision["crop"], "money": farm.get("money", 0)})
        return {"farmer": farmer, "hands": [["PASS"] for _ in farm.get("hands", [])], "market": market}
    return agent

def _get_brain():
    global _BRAIN
    if _BRAIN is None:
        _BRAIN = FlyBrain()
    return _BRAIN

agent = make_agent()
