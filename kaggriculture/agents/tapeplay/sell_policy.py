
# ----------------------------------------------------------------------------- adaptive selling
import math as _tp_math

_TP_MARKET = {
    "WHEAT": (25, 400, "sqrt", 0.80, "log", 0.20), "CARROT": (35, 450, "hinge", 1.00, "sqrt", 0.70),
    "TOMATO": (60, 200, "hinge", 0.40, "sqrt", 0.60), "STRAWBERRY": (120, 100, "sqrt", 0.70, "linear", 1.60),
    "MELON": (250, 300, "log", 0.20, "sq", 3.60), "EGG": (50, 332, "hinge", 0.40, "log", 0.20),
    "MILK": (160, 122, "sqrt", 0.60, "linear", 1.60), "WOOL": (200, 105, "log", 0.20, "sq", 3.20),
    "FERTILIZER": (100, 200, "linear", 0.40, "linear", 0.40),
}
_TP_SHOPS = {
    "BAKERY": ["EGG", "WHEAT"], "PIZZA_SHOP": ["MILK", "TOMATO", "WHEAT"],
    "BRUNCH_SPOT": ["EGG", "WHEAT", "STRAWBERRY"], "YARN_STORE": ["WOOL"],
    "ICE_CREAM_SHOP": ["STRAWBERRY", "MILK", "WHEAT"], "PET_CAFE": ["CARROT"],
    "SMOOTHIE_SHOP": ["STRAWBERRY", "MILK"], "FARMERS_MARKET": ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY"],
}
_TP_PREMIUM = ("STRAWBERRY", "MILK", "WOOL", "MELON")
_TP_LAST = 718
TP = {
    "frac": {"STRAWBERRY": 0.95, "MILK": 0.5, "WOOL": 0.5, "MELON": 0.75, "TOMATO": 0.8, "CARROT": 0.7,
             "EGG": 0.6, "WHEAT": 0.7, "FERTILIZER": 0.3},
    "lot": {"STRAWBERRY": 4, "MILK": 4, "WOOL": 4, "MELON": 3, "TOMATO": 4, "CARROT": 8, "EGG": 6, "WHEAT": 8,
            "FERTILIZER": 3},
    "reserve_horizon": 30,     # steps of future tape pickups to keep in the shed
    "shed_soft_cap": 55,
}


def _tp_shape(f, x, T):
    x = max(0.0, x)
    if f == "linear":
        return x
    if f == "sq":
        return x * x
    if f == "sqrt":
        return _tp_math.sqrt(x)
    if f == "log":
        return _tp_math.log(1.0 + x)
    if f == "hinge":
        u = x / T
        return u + 8.0 * max(0.0, u - 1.0) ** 2
    return x


def _tp_price(item, inv):
    base, T, bf, bt, af, at = _TP_MARKET[item]
    if inv < 10000:
        p = base + bt * base / _tp_shape(bf, T, T) * _tp_shape(bf, 10000 - inv, T)
    else:
        p = base - at * base / _tp_shape(af, T, T) * _tp_shape(af, inv - 10000, T)
    return max(1, int(round(p)))


def _tp_demand_per_day(shops, item):
    d = 1.0 if item != "FERTILIZER" else 0.0
    for s in shops:
        prods = _TP_SHOPS.get(s, [])
        if item in prods:
            d += 6.0 * (2 if len(prods) == 1 else 1)
    return d


def _tp_future_pickups(step, horizon):
    need = {}
    for s in range(step, min(len(_TP_ACTS), step + horizon)):
        f, hands, _m = _TP_ACTS[s]
        for u in [f] + list(hands):
            if u and u[0] == "PICKUP" and len(u) >= 2:
                need[u[1]] = need.get(u[1], 0) + (int(u[2]) if len(u) >= 3 else 1)
    return need


def _tp_sells(obs, step, pick_now):
    day, hour = step // 24, step % 24
    shed = dict(obs["private"]["shed"])
    for k, v in pick_now.items():
        shed[k] = shed.get(k, 0) - v
    minv = dict(obs["market"]["inventory"])
    shops = list(obs["town"]["unlocked_shops"])
    tick = step % 4 == 1
    days_left = (_TP_LAST - step) / 24.0
    final = step >= _TP_LAST - 1
    reserve = _tp_future_pickups(step + 1, TP["reserve_horizon"])
    shed_total = sum(v for v in shed.values() if v > 0)
    premium, other = [], []
    for item in ("STRAWBERRY", "MILK", "WOOL", "MELON", "TOMATO", "EGG", "CARROT", "WHEAT", "FERTILIZER"):
        have = shed.get(item, 0) - (0 if final else reserve.get(item, 0))
        if have <= 0:
            continue
        if final:
            n = have
        else:
            if item in _TP_PREMIUM and not tick and days_left > 1.0:
                continue
            base = _TP_MARKET[item][0]
            frac = TP["frac"][item]
            lot = TP["lot"][item]
            absorb = _tp_demand_per_day(shops, item) * max(0.0, days_left - 0.5)
            if have > absorb + 6 or item == "FERTILIZER":
                frac *= 0.3
                lot += int(min(12, have - absorb))
            if days_left < 1.0:
                frac *= 0.25
                lot = max(lot, int(_tp_math.ceil(have / max(1.0, days_left * 6))))
            elif days_left < 3:
                frac *= 0.7
                lot += 2
            if shed_total > TP["shed_soft_cap"]:
                frac *= 0.5
                lot += 8
            thr = max(2, frac * base)
            n = 0
            inv = minv[item]
            while n < min(have, lot) and _tp_price(item, inv + n) >= thr:
                n += 1
        if n > 0:
            (premium if item in _TP_PREMIUM else other).append(["SELL", item, int(n)])
    return premium, other


def _tp_expected_quads(step):
    """Quadrants the recorded player owned by `step`: each run of BUY_LAND orders ends in a purchase."""
    n, in_run = 1, False
    for s in range(0, min(step, len(_TP_ACTS))):
        has = any(o and o[0] == "BUY_LAND" for o in _TP_ACTS[s][2])
        if in_run and not has:
            n += 1
        in_run = has
    return n


_TP_QUADS = {}



def _cg_fib_sum(n):
    a, b, s = 1, 1, 0
    for _ in range(n):
        s += a
        a, b = b, a + b
    return s


_CG_COST = {"WHEAT": 10, "CARROT": 20, "TOMATO": 50, "STRAWBERRY": 100, "MELON": 80}


def _cg_guard(obs, step, acts, market):
    """Keep enough cash for the tape's next hour-0 hires (the opening leaves ~$5; a failed hire derails the tape)."""
    try:
        hour = step % 24
        me = obs["player"]
        money = float(obs["farms"][me]["money"])
        shed = obs["private"]["shed"]
        prices = obs["market"]["prices"]
        if hour == 0:
            need = _cg_fib_sum(sum(1 for o in market if o and o[0] == "HIRE"))
            other = sum(1 for o in market if o and o[0] == "SELL")
            if money < need and not other:
                pre = []
                for item in ("FERTILIZER", "WHEAT", "EGG", "MILK"):
                    if money >= need:
                        break
                    k = 0
                    while k < shed.get(item, 0) and money < need:
                        k += 1
                        money += max(1, prices.get(item, 1))
                    if k:
                        pre.append(["SELL", item, k])
                return pre + list(market)
            return market
        if hour < 16:
            return market
        nxt = (step // 24 + 1) * 24
        if nxt >= len(acts):
            return market
        need = _cg_fib_sum(sum(1 for o in acts[nxt][2] if o and o[0] == "HIRE")) + 2
        spend = 0
        for o in market:
            if o and o[0] == "BUY_SEED":
                spend += _CG_COST.get(o[1], 10) * int(o[2])
        if money - spend >= need:
            return market
        out = []
        for o in market:
            if o and o[0] == "BUY_SEED" and money - spend < need:
                spend -= _CG_COST.get(o[1], 10) * int(o[2])
                continue
            out.append(o)
        return out
    except Exception:
        return market


def tapeplay_agent(obs, config=None):
    step = obs["step"] if "step" in obs else obs["day"] * 24 + obs["hour"]
    me = obs["player"]
    n_h = len(obs["farms"][me]["hands"])
    if step >= len(_TP_ACTS):
        return {"farmer": ["PASS"], "hands": [["PASS"]] * n_h, "market": []}
    f, hands, market = _TP_ACTS[step]
    hands = (list(hands) + [["PASS"]] * n_h)[:n_h]
    day = step // 24
    market = list(market)
    # catch up on land the recorded player already owned at this point
    q_have = len(obs["farms"][me]["unlocked_quadrants"])
    if step not in _TP_QUADS:
        _TP_QUADS[step] = _tp_expected_quads(step)
    if q_have < _TP_QUADS[step] and not any(o and o[0] == "BUY_LAND" for o in market):
        market = [["BUY_LAND"]] + market
    market = _cg_guard(obs, step, _TP_ACTS, market)
    if day < _TP_SWITCH_DAY:
        return {"farmer": f, "hands": hands, "market": market[:10]}
    try:
        pick_now = {}
        for u in [f] + hands:
            if u and u[0] == "PICKUP" and len(u) >= 2:
                pick_now[u[1]] = pick_now.get(u[1], 0) + (int(u[2]) if len(u) >= 3 else 1)
        premium, other = _tp_sells(obs, step, pick_now)
        buys = [o for o in market if o and o[0] != "SELL"]
        room = max(0, 10 - len(buys))
        mk = buys + (premium + other)[:room]
    except Exception:
        mk = market
    return {"farmer": f, "hands": hands, "market": mk}
