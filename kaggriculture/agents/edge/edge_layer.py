

# ===========================================================================
# EDGE: market/sell-side post-processing layer (appended to a public base).
# The base's exported entry is captured below; `edge_agent` must stay the
# LAST callable defined in this file (Kaggle picks the last callable).
# ===========================================================================
import math as _edge_math
import itertools as _edge_it

_EDGE_BASE = [v for v in list(globals().values()) if callable(v)][-1]
_EDGE_CFG = dict(__EDGE_CFG__)

_EDGE_P = {'WHEAT': {'base': 25, 'I0': 10000, 'T': 400, 'below_func': 'sqrt', 'below_target': 0.8, 'above_func': 'log', 'above_target': 0.2}, 'CARROT': {'base': 35, 'I0': 10000, 'T': 450, 'below_func': 'hinge', 'below_target': 1.0, 'above_func': 'sqrt', 'above_target': 0.7}, 'TOMATO': {'base': 60, 'I0': 10000, 'T': 200, 'below_func': 'hinge', 'below_target': 0.4, 'above_func': 'sqrt', 'above_target': 0.6}, 'STRAWBERRY': {'base': 120, 'I0': 10000, 'T': 100, 'below_func': 'sqrt', 'below_target': 0.7, 'above_func': 'linear', 'above_target': 1.6}, 'MELON': {'base': 250, 'I0': 10000, 'T': 300, 'below_func': 'log', 'below_target': 0.2, 'above_func': 'sq', 'above_target': 3.6}, 'EGG': {'base': 50, 'I0': 10000, 'T': 332, 'below_func': 'hinge', 'below_target': 0.4, 'above_func': 'log', 'above_target': 0.2}, 'MILK': {'base': 160, 'I0': 10000, 'T': 122, 'below_func': 'sqrt', 'below_target': 0.6, 'above_func': 'linear', 'above_target': 1.6}, 'WOOL': {'base': 200, 'I0': 10000, 'T': 105, 'below_func': 'log', 'below_target': 0.2, 'above_func': 'sq', 'above_target': 3.2}, 'FERTILIZER': {'base': 100, 'I0': 10000, 'T': 200, 'below_func': 'linear', 'below_target': 0.4, 'above_func': 'linear', 'above_target': 0.4}}
_EDGE_ANIMAL = {"COW": 400, "SHEEP": 500, "GOOSE": 300}
_EDGE_SEED = {"WHEAT": 10, "CARROT": 20, "TOMATO": 50, "STRAWBERRY": 100, "MELON": 80}
_EDGE_LAND = (1000, 2000, 4000)
_EDGE_SHOPS = {"BAKERY": ("EGG", "WHEAT"), "PIZZA_SHOP": ("MILK", "TOMATO", "WHEAT"),
               "BRUNCH_SPOT": ("EGG", "WHEAT", "STRAWBERRY"), "YARN_STORE": ("WOOL",),
               "ICE_CREAM_SHOP": ("STRAWBERRY", "MILK", "WHEAT"), "PET_CAFE": ("CARROT",),
               "SMOOTHIE_SHOP": ("STRAWBERRY", "MILK"), "FARMERS_MARKET": ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY")}
_EDGE_STATE = {}
_EDGE_REPORT = dict(edge_sa_turns=0, edge_sa_units=0, edge_l2_turns=0, edge_l2_gain=0.0, edge_errors=0)


def _edge_shape(func, x, T):
    x = max(0.0, x)
    if func == "linear": return x
    if func == "sq": return x * x
    if func == "sqrt": return _edge_math.sqrt(x)
    if func == "log": return _edge_math.log(1.0 + x)
    if func == "hinge":
        u = x / T
        return u + 8.0 * max(0.0, u - 1.0) ** 2
    return x


def _edge_params(obs):
    params = {k: dict(v) for k, v in _EDGE_P.items()}
    for k, patch in ((obs.get('market') or {}).get('params') or {}).items():
        if k in params and isinstance(patch, dict):
            params[k].update(patch)
    return params


def _edge_price(item, inv, params):
    p = params[item]
    base, I0, T = p["base"], p["I0"], p["T"]
    if inv < I0:
        f = p["below_func"]; amp = p["below_target"] * base / _edge_shape(f, T, T)
        price = base + amp * _edge_shape(f, I0 - inv, T)
    else:
        f = p["above_func"]; amp = p["above_target"] * base / _edge_shape(f, T, T)
        price = base - amp * _edge_shape(f, inv - I0, T)
    return max(1, int(round(price)))


def _edge_drain(step, shops):
    out = {}
    if step % 4 == 0:
        for s in shops:
            items = _EDGE_SHOPS.get(s, ())
            for it in items:
                out[it] = out.get(it, 0) + (2 if len(items) == 1 else 1)
    if step % 24 == 0:
        for it in _EDGE_P:
            if it != "FERTILIZER":
                out[it] = out.get(it, 0) + 1
    return out


def _edge_proj(obs, action):
    try:
        return {k: max(0, int(v)) for k, v in projected_shed(action, FarmView(obs)).items()}
    except Exception:
        return {k: max(0, int(v)) for k, v in dict(obs['private']['shed']).items()}


def _edge_sold(market):
    out = {}
    for o in market:
        if o and len(o) >= 3 and o[0] == 'SELL':
            out[o[1]] = out.get(o[1], 0) + max(0, int(o[2]))
    return out


_EDGE_PREM = ('STRAWBERRY', 'MELON', 'MILK', 'WOOL', 'EGG', 'TOMATO', 'CARROT')


def _edge_track(obs, st):
    """Infer the rival's premium sales of the previous turn; count pre-emptions (rival sold X while we held X
    and our base did not want to sell X)."""
    prev = st.get('prev')
    step = int(obs['step'])
    if not prev or prev['step'] != step - 1:
        return
    inv = obs['market']['inventory']
    drain = _edge_drain(prev['step'], prev['shops'])
    pre = st.setdefault('pre', {})
    for item in _EDGE_PREM:
        opp = int(inv[item]) - prev['inv'][item] + drain.get(item, 0) - prev['added'].get(item, 0)
        if opp > 0:
            st.setdefault('opp_units', {})[item] = st.get('opp_units', {}).get(item, 0) + opp
            if prev['held'].get(item, 0) > 0 and item not in prev['base_sells']:
                pre[item] = pre.get(item, 0) + 1
                _EDGE_REPORT['edge_pre_events'] = _EDGE_REPORT.get('edge_pre_events', 0) + 1


def _edge_remember(obs, st, base_market, out):
    market = out.get('market') or []
    stock = _edge_proj(obs, out)
    params = _edge_params(obs)
    inv = {k: int(v) for k, v in obs['market']['inventory'].items()}
    left = dict(stock)
    added = {}
    for o in market:
        if o and len(o) >= 3 and o[0] == 'SELL' and o[1] in params:
            item = o[1]
            n = min(max(0, int(o[2])), left.get(item, 0))
            for _ in range(n):
                if _edge_price(item, inv[item], params) > 1:
                    inv[item] += 1
                    added[item] = added.get(item, 0) + 1
            left[item] = left.get(item, 0) - n
    st['prev'] = dict(step=int(obs['step']), inv={k: int(v) for k, v in obs['market']['inventory'].items()},
                      shops=list((obs.get('town') or {}).get('unlocked_shops') or []), added=added,
                      held={k: v for k, v in left.items() if v > 0},
                      base_sells={o[1] for o in base_market if o and len(o) >= 3 and o[0] == 'SELL' and int(o[2]) > 0})


def _edge_sell_ahead(obs, action, st):
    cfg = _EDGE_CFG
    step = int(obs['step'])
    if not (cfg['sa_from'] <= step <= cfg['sa_to']):
        return action
    if (step % 4) not in cfg['sa_mod4']:
        return action
    market = [list(o) if isinstance(o, (list, tuple)) else o for o in (action.get('market') or [])]
    stock = _edge_proj(obs, action)
    sold = _edge_sold(market)
    params = _edge_params(obs)
    inv = obs['market']['inventory']
    added = []
    for item in cfg['sa_items']:
        rem = stock.get(item, 0) - sold.get(item, 0)
        if rem <= 0:
            continue
        # Held-stock only: skip units that just landed this turn unless allowed.
        if cfg.get('sa_held_only'):
            prev = st.get('shed_prev', {}).get(item, 0)
            rem = min(rem, prev - sold.get(item, 0)) if prev > 0 else 0
            if rem <= 0:
                continue
        pb = params[item]['base']
        q = 0
        cur = int(inv[item]) + sold.get(item, 0)
        qmax = (cfg.get('sa_max_item') or {}).get(item, cfg['sa_max'])
        if cfg.get('agg') and st.get('pre', {}).get(item if cfg.get('agg_per_item', True) else '', 0) >= cfg['agg_k']:
            qmax = cfg['agg_max']
        elif cfg.get('agg') and not cfg.get('agg_per_item', True) and sum(st.get('pre', {}).values()) >= cfg['agg_k']:
            qmax = cfg['agg_max']
        ratio = (cfg.get('sa_ratio_item') or {}).get(item, cfg['sa_min_ratio'])
        while q < rem and q < qmax:
            pr = _edge_price(item, cur + q, params)
            if pr < ratio * pb or pr <= cfg['sa_min_abs']:
                break
            q += 1
        if q <= 0:
            continue
        added.append((item, q))
    if not added:
        return action
    for item, q in added:
        for o in market:
            if o and len(o) >= 3 and o[0] == 'SELL' and o[1] == item:
                o[2] = int(o[2]) + q
                break
        else:
            # Never push a base order out of the 10 processed slots.
            if len(market) >= 10:
                holes = [i for i, o in enumerate(market) if not o]
                if not holes:
                    continue
                del market[holes[-1]]
            pos = cfg.get('sa_pos') or ('front' if cfg['sa_front'] else 'end')
            if pos == 'front':
                market.insert(0, ['SELL', item, q])
            elif pos == 'after_sells':
                k = 0
                while k < len(market) and market[k] and market[k][0] == 'SELL':
                    k += 1
                if k < len(market) and not market[k]:
                    market[k] = ['SELL', item, q]
                else:
                    market.insert(k, ['SELL', item, q])
            else:
                market.append(['SELL', item, q])
        _EDGE_REPORT['edge_sa_units'] += q
    _EDGE_REPORT['edge_sa_turns'] += 1
    return dict(action, market=market[:10])


def _edge_floor_hold(obs, action):
    """Trim SELL units that would fill at or below fh_px (own-queue estimate) before fh_until."""
    cfg = _EDGE_CFG
    step = int(obs['step'])
    if step >= cfg['fh_until']:
        return action
    market = [list(o) if isinstance(o, (list, tuple)) else o for o in (action.get('market') or [])]
    stock = _edge_proj(obs, action)
    shed_total = sum(stock.values())
    params = _edge_params(obs)
    inv = {k: int(v) for k, v in obs['market']['inventory'].items()}
    changed = False
    kept_total = 0
    sold_all = 0
    left = dict(stock)
    for o in market:
        if o and len(o) >= 3 and o[0] == 'SELL':
            k = min(max(0, int(o[2])), left.get(o[1], 0)); left[o[1]] = left.get(o[1], 0) - k; sold_all += k
    leftover = shed_total - sold_all
    for o in market:
        if not (o and len(o) >= 3 and o[0] == 'SELL' and o[1] in cfg['fh_items']):
            continue
        item = o[1]
        n = min(max(0, int(o[2])), stock.get(item, 0))
        q = 0
        while q < n and _edge_price(item, inv[item] + q, params) > cfg['fh_px']:
            q += 1
        keep = n - q
        if keep > 0 and leftover + kept_total + keep <= cfg['fh_shed']:
            o[2] = q
            kept_total += keep
            changed = True
        inv[item] += q
        stock[item] = stock.get(item, 0) - q
    if not changed:
        return action
    market = [o if not (o and o[0] == 'SELL' and int(o[2]) <= 0) else [] for o in market]
    _EDGE_REPORT['edge_fh_turns'] = _EDGE_REPORT.get('edge_fh_turns', 0) + 1
    return dict(action, market=market)


def _edge_margin_fn(opp, inv0, stock, params):
    return _v44y_factor_margin(opp, inv0, stock, params)


def _edge_level2(obs, action, model=None):
    """Best response of our SELL ordering against a rival who submits `model` (default: our own final orders)."""
    market = [list(o) if isinstance(o, (list, tuple)) else o for o in (action.get('market') or [])]
    sells = [(i, o) for i, o in enumerate(market) if o and o[0] == 'SELL']
    if len(sells) < 2 or len(sells) > 6:
        return action
    bought = {o[1] for o in market if o and len(o) > 1 and o[0] == 'BUY_PRODUCT'}
    if any(o[1] in bought for _, o in sells):
        return action
    slots = [i for i, _ in sells]
    orders = [o for _, o in sells]
    stock = _edge_proj(obs, action)
    params = _v44y_params(obs)
    inv0 = {k: int(v) for k, v in obs['market']['inventory'].items()}
    opp = [list(o) if isinstance(o, (list, tuple)) else o for o in (model if model is not None else market)]
    margin = _edge_margin_fn(opp, inv0, stock, params)
    base = best = margin(market)
    best_m = None
    for perm in _edge_it.permutations(range(len(orders))):
        cand = list(market)
        for s, p in zip(slots, perm):
            cand[s] = orders[p]
        if cand == market:
            continue
        v = margin(cand)
        if v > best + 0.5:
            best, best_m = v, cand
    if best_m is None:
        return action
    _EDGE_REPORT['edge_l2_turns'] += 1
    _EDGE_REPORT['edge_l2_gain'] += best - base
    return dict(action, market=best_m)


def _edge_sells_first(obs, action):
    """Move SELL orders ahead of HIRE/BUY_* when current cash already covers every purchase."""
    market = [list(o) if isinstance(o, (list, tuple)) else o for o in (action.get('market') or [])]
    if len(market) < 2:
        return action
    seat = int(obs['player']); farm = obs['farms'][seat]
    cash = float(farm.get('money', 0) or 0)
    prices = obs['market'].get('prices') or {}
    cost = 0.0; quads = len(farm.get('unlocked_quadrants') or [])
    hires = int(farm.get('hires_today', 0) or 0); nh = len(farm.get('hands') or [])
    for o in market:
        if not o: continue
        if o[0] == 'BUY_ANIMAL': cost += _EDGE_ANIMAL.get(o[1], 500) * max(0, int(o[2]))
        elif o[0] == 'BUY_SEED': cost += _EDGE_SEED.get(o[1], 100) * max(0, int(o[2]))
        elif o[0] == 'BUY_PRODUCT': cost += 2.0 * float(prices.get(o[1], 0)) * max(0, int(o[2])) + 50
        elif o[0] == 'BUY_LAND': cost += 4000
        elif o[0] == 'HIRE': cost += 200
    if cost > cash:
        return action
    bought = {o[1] for o in market if o and len(o) > 1 and o[0] == 'BUY_PRODUCT'}
    sells = [o for o in market if o and o[0] == 'SELL' and o[1] not in bought]
    if not sells:
        return action
    rest = [o for o in market if not (o and o[0] == 'SELL' and o[1] not in bought)]
    new = sells + rest
    if new == market:
        return action
    return dict(action, market=new)


def edge_agent(observation, configuration=None):
    action = _EDGE_BASE(observation, configuration)
    try:
        step = int(observation['step']); seat = int(observation['player'])
        st = _EDGE_STATE.get(seat)
        if st is None or step <= st.get('step', -1):
            st = _EDGE_STATE[seat] = {'step': -1, 'shed_prev': {}}
            if step == 0:
                for k in _EDGE_REPORT: _EDGE_REPORT[k] = 0
        st['step'] = step
        if not isinstance(action, dict):
            return action
        # Clone gate: the rival's unit positions and farm layout match ours.
        try:
            farms = observation['farms']
            own, riv = farms[seat], farms[1 - seat]
            eq = bool(own.get('hands')) and own['hands'] == riv['hands'] and own['farmer'] == riv['farmer']
            sim = _r37_similarity(observation) if eq else 0.0
            hist = st.setdefault('clone_hist', [])
            hist.append(1 if (eq and sim >= 0.95) else 0)
            if len(hist) > _EDGE_CFG.get('gate_win', 24):
                hist.pop(0)
            st['clone'] = len(hist) >= 4 and sum(hist) >= _EDGE_CFG.get('gate_frac', 0.5) * len(hist)
            if st['clone']:
                st['clone_seen'] = True
        except Exception:
            st['clone'] = False
        try:
            _edge_track(observation, st)
        except Exception:
            _EDGE_REPORT['edge_errors'] += 1
        out = action
        base_market = [list(o) if isinstance(o, (list, tuple)) else o for o in (action.get('market') or [])]
        gate = _EDGE_CFG.get('sa_gate')
        gate_ok = (gate is None or (gate == 'clone' and st.get('clone'))
                   or (gate == 'clone_sticky' and st.get('clone_seen')))
        if _EDGE_CFG['sa'] and gate_ok:
            _EDGE_REPORT['edge_gate_turns'] = _EDGE_REPORT.get('edge_gate_turns', 0) + 1
            out = _edge_sell_ahead(observation, out, st)
        if _EDGE_CFG.get('fh'):
            out = _edge_floor_hold(observation, out)
        if _EDGE_CFG['sells_first']:
            out = _edge_sells_first(observation, out)
        if _EDGE_CFG['l2'] and step >= _EDGE_CFG['l2_from']:
            out = _edge_level2(observation, out, base_market if _EDGE_CFG.get('l2_model') == 'base' else None)
        try:
            _edge_remember(observation, st, base_market, out)
        except Exception:
            _EDGE_REPORT['edge_errors'] += 1
        # remember the post-market stock we intend to keep (for held-only logic)
        stock = _edge_proj(observation, out)
        sold = _edge_sold(out.get('market') or [])
        st['shed_prev'] = {k: max(0, v - sold.get(k, 0)) for k, v in stock.items()}
        return out
    except Exception:
        _EDGE_REPORT['edge_errors'] += 1
        return action


edge_agent.telemetry = _EDGE_REPORT
