
import base64 as _rt_b64
import gzip as _rt_gz
import hashlib as _rt_hash
import json as _rt_json
from collections import Counter as _RtCounter

_RT = _rt_json.loads(_rt_gz.decompress(_rt_b64.b64decode(_RT_BLOB)).decode())
del _RT_BLOB
RT = {"w_pos": 2.0, "w_overlap": 1.0, "w_prior": 4.0, "w_won": 0.5, "stick": 0.3, "dmax": 4, "w_dist": 1.5}
_RT_CODE = {"WHEAT": "w", "CARROT": "c", "TOMATO": "t", "STRAWBERRY": "s", "MELON": "m", "COW": "C", "SHEEP": "S", "GOOSE": "G"}


def _rt_lay(farm):
    out = []
    for row in farm["tiles"]:
        for t in row:
            if t is None:
                out.append(".")
            elif t == "LOCKED":
                out.append("#")
            elif t.get("kind") == "WEED":
                out.append("x")
            elif t.get("kind") == "PLANT":
                out.append(_RT_CODE.get(t.get("crop"), "?"))
            elif t.get("animal"):
                out.append(_RT_CODE.get(t.get("animal"), "?"))
            else:
                out.append("P" if t.get("kind") == "PASTURE" else "O")
    return "".join(out)


def _rt_dist(a, b):
    d = 0
    for x, y in zip(a, b):
        if x != y and not ({x, y} <= {".", "x"}):
            d += 1
    return d


def _rt_sig(farm):
    parts = []
    for y, row in enumerate(farm["tiles"]):
        for x, t in enumerate(row):
            if isinstance(t, dict):
                parts.append(f"{x}{y}{t.get('kind', '')[:2]}{(t.get('crop') or t.get('animal') or '')[:2]}")
    parts.append("|" + ",".join(sorted(farm.get("unlocked_quadrants") or [])))
    return _rt_hash.md5(";".join(parts).encode()).hexdigest()[:12]


def _rt_expected_quads(acts, step):
    n, in_run = 1, False
    for s in range(0, min(step, len(acts))):
        has = any(o and o[0] == "BUY_LAND" for o in acts[s][2])
        if in_run and not has:
            n += 1
        in_run = has
    return n


def _rt_default():
    # opening: most robust tape inside the largest day-1 layout group
    groups = {}
    for tid, v in _RT.items():
        s = v["l"].get("1")
        if s:
            groups.setdefault(s, []).append(tid)
    big = max(groups.values(), key=len) if groups else list(_RT)
    return max(big, key=lambda t: (_RT[t]["p"], _RT[t]["w"]))


_RT_STATE = {"tape": None, "step": -1, "log": [], "quads": {}}


def _rt_choose(day, farm, shops, current):
    lay = _rt_lay(farm)
    dist = {}
    for tid, v in _RT.items():
        tl = v["l"].get(str(day))
        if tl:
            dist[tid] = _rt_dist(lay, tl)
    if not dist:
        return current, 0
    dmin = min(dist.values())
    if dmin > RT["dmax"]:
        return current, 0
    cands = [t for t, d in dist.items() if d <= dmin + 1]
    best, best_sc = current, None
    for tid in cands:
        v = _RT[tid]
        ts = v["sh"].get(str(day)) or []
        pos = sum(1 for i in range(min(len(ts), len(shops))) if ts[i] == shops[i])
        ov = sum((_RtCounter(ts) & _RtCounter(shops)).values())
        sc = (RT["w_pos"] * pos + RT["w_overlap"] * ov + RT["w_prior"] * v["p"] + RT["w_won"] * v["w"]
              - RT["w_dist"] * dist[tid])
        if tid == current:
            sc += RT["stick"]
        if best_sc is None or sc > best_sc:
            best, best_sc = tid, sc
    return best, len(cands)



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


def router_agent(obs, config=None):
    step = obs["step"] if "step" in obs else obs["day"] * 24 + obs["hour"]
    me = obs["player"]
    farm = obs["farms"][me]
    st = _RT_STATE
    if step == 0 or step < st["step"] or st["tape"] is None:
        st["tape"] = _rt_default()
        st["log"] = []
        st["quads"] = {}
    st["step"] = step
    n_h = len(farm["hands"])
    try:
        day, hour = step // 24, step % 24
        if hour == 0 and 1 <= day <= _RT_MAX_DAY:
            new, nc = _rt_choose(day, farm, list(obs["town"]["unlocked_shops"]), st["tape"])
            if new != st["tape"]:
                st["log"].append((step, st["tape"], new, nc))
                st["tape"] = new
        acts = _RT[st["tape"]]["a"]
        if step >= len(acts):
            return {"farmer": ["PASS"], "hands": [["PASS"]] * n_h, "market": []}
        f, hands, market = acts[step]
        hands = (list(hands) + [["PASS"]] * n_h)[:n_h]
        market = list(market)
        key = (st["tape"], step)
        if key not in st["quads"]:
            st["quads"] = {key: _rt_expected_quads(acts, step)}
        if len(farm["unlocked_quadrants"]) < st["quads"][key] and not any(o and o[0] == "BUY_LAND" for o in market):
            market = [["BUY_LAND"]] + market
        market = _cg_guard(obs, step, acts, market)
        return {"farmer": f, "hands": hands, "market": market[:10]}
    except Exception:
        return {"farmer": ["PASS"], "hands": [["PASS"]] * n_h, "market": []}
