

# ===========================================================================
# TAPE EDGE: market-only post-processing for a replay-tape base (standalone).
# Unit actions and purchase orders of the tape are never changed or dropped.
# `tape_edge_agent` must stay the LAST callable defined in this file.
# ===========================================================================
import math as _te_math
import itertools as _te_it

_TE_BASE = [v for v in list(globals().values()) if callable(v)][-1]
_TE_CFG = dict(__TE_CFG__)
_TE_P = {'WHEAT': {'base': 25, 'I0': 10000, 'T': 400, 'below_func': 'sqrt', 'below_target': 0.8, 'above_func': 'log', 'above_target': 0.2}, 'CARROT': {'base': 35, 'I0': 10000, 'T': 450, 'below_func': 'hinge', 'below_target': 1.0, 'above_func': 'sqrt', 'above_target': 0.7}, 'TOMATO': {'base': 60, 'I0': 10000, 'T': 200, 'below_func': 'hinge', 'below_target': 0.4, 'above_func': 'sqrt', 'above_target': 0.6}, 'STRAWBERRY': {'base': 120, 'I0': 10000, 'T': 100, 'below_func': 'sqrt', 'below_target': 0.7, 'above_func': 'linear', 'above_target': 1.6}, 'MELON': {'base': 250, 'I0': 10000, 'T': 300, 'below_func': 'log', 'below_target': 0.2, 'above_func': 'sq', 'above_target': 3.6}, 'EGG': {'base': 50, 'I0': 10000, 'T': 332, 'below_func': 'hinge', 'below_target': 0.4, 'above_func': 'log', 'above_target': 0.2}, 'MILK': {'base': 160, 'I0': 10000, 'T': 122, 'below_func': 'sqrt', 'below_target': 0.6, 'above_func': 'linear', 'above_target': 1.6}, 'WOOL': {'base': 200, 'I0': 10000, 'T': 105, 'below_func': 'log', 'below_target': 0.2, 'above_func': 'sq', 'above_target': 3.2}, 'FERTILIZER': {'base': 100, 'I0': 10000, 'T': 200, 'below_func': 'linear', 'below_target': 0.4, 'above_func': 'linear', 'above_target': 0.4}}
_TE_SHOPS = {"BAKERY": ("EGG", "WHEAT"), "PIZZA_SHOP": ("MILK", "TOMATO", "WHEAT"),
             "BRUNCH_SPOT": ("EGG", "WHEAT", "STRAWBERRY"), "YARN_STORE": ("WOOL",),
             "ICE_CREAM_SHOP": ("STRAWBERRY", "MILK", "WHEAT"), "PET_CAFE": ("CARROT",),
             "SMOOTHIE_SHOP": ("STRAWBERRY", "MILK"), "FARMERS_MARKET": ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY")}
_TE_PREM = ('STRAWBERRY', 'MELON', 'MILK', 'WOOL', 'EGG', 'TOMATO', 'CARROT')
_TE_ALL = ('WHEAT', 'CARROT', 'TOMATO', 'STRAWBERRY', 'MELON', 'EGG', 'MILK', 'WOOL', 'FERTILIZER')
_TE_ANIMALS = ('COW', 'SHEEP', 'GOOSE')
_TE_STATE = {}
_TE_REPORT = dict(sa_units=0, l2_turns=0, front_turns=0, fg_units=0, liq_units=0, errors=0, armed_at=-1)


def _te_shape(func, x, T):
    x = max(0.0, x)
    if func == "linear": return x
    if func == "sq": return x * x
    if func == "sqrt": return _te_math.sqrt(x)
    if func == "log": return _te_math.log(1.0 + x)
    if func == "hinge":
        u = x / T
        return u + 8.0 * max(0.0, u - 1.0) ** 2
    return x


def _te_params(obs):
    params = {k: dict(v) for k, v in _TE_P.items()}
    for k, patch in ((obs.get('market') or {}).get('params') or {}).items():
        if k in params and isinstance(patch, dict):
            params[k].update(patch)
    return params


def _te_price(item, inv, params):
    p = params[item]
    base, I0, T = p["base"], p["I0"], p["T"]
    if inv < I0:
        f = p["below_func"]; amp = p["below_target"] * base / _te_shape(f, T, T)
        price = base + amp * _te_shape(f, I0 - inv, T)
    else:
        f = p["above_func"]; amp = p["above_target"] * base / _te_shape(f, T, T)
        price = base - amp * _te_shape(f, inv - I0, T)
    return max(1, int(round(price)))


def _te_drain(step, shops):
    out = {}
    if step % 4 == 0:
        for s in shops:
            items = _TE_SHOPS.get(s, ())
            for it in items:
                out[it] = out.get(it, 0) + (2 if len(items) == 1 else 1)
    if step % 24 == 0:
        for it in _TE_P:
            if it != "FERTILIZER":
                out[it] = out.get(it, 0) + 1
    return out


def _te_adjacent(pos, board):
    half = board // 2
    try:
        return pos[0] in (half - 1, half) and pos[1] in (half - 1, half)
    except Exception:
        return False


def _te_proj(obs, action, cap=100):
    """Shed after this turn's unit actions, before the market."""
    seat = int(obs['player'])
    farm = obs['farms'][seat]
    priv = obs.get('private') or {}
    shed = {k: max(0, int(v)) for k, v in dict(priv.get('shed') or {}).items()}
    invs = [dict(i or {}) for i in (priv.get('inventories') or [])]
    board = len(farm.get('tiles') or []) or 10
    positions = [farm.get('farmer')] + [list(p) for p in (farm.get('hands') or [])]
    units = [action.get('farmer') or ['PASS']] + list(action.get('hands') or [])
    total = sum(shed.values())
    for i in range(min(len(units), len(positions))):
        act = units[i] or ['PASS']
        if not _te_adjacent(positions[i], board):
            continue
        op = act[0]
        inv = invs[i] if i < len(invs) else {}
        if op == 'PICKUP' and len(act) >= 2:
            n = int(act[2]) if len(act) >= 3 else 1
            n = min(max(0, n), shed.get(act[1], 0))
            shed[act[1]] = shed.get(act[1], 0) - n; total -= n
        elif op == 'DROP':
            for item, held in inv.items():
                take = min(max(0, int(held)), max(0, cap - total))
                if take > 0:
                    shed[item] = shed.get(item, 0) + take; total += take
        elif op == 'PLACE' and len(act) >= 2 and act[1] not in _TE_ANIMALS:
            n = int(act[2]) if len(act) >= 3 else 1
            take = min(max(0, n), max(0, int(inv.get(act[1], 0))), max(0, cap - total))
            if take > 0:
                shed[act[1]] = shed.get(act[1], 0) + take; total += take
    return shed


def _te_sold(market):
    out = {}
    for o in market:
        if o and len(o) >= 3 and o[0] == 'SELL':
            out[o[1]] = out.get(o[1], 0) + max(0, int(o[2]))
    return out


def _te_future_pickups(step, horizon):
    need = {}
    acts = globals().get('_TP_ACTS') or []
    for s in range(step, min(len(acts), step + horizon)):
        f, hands, _m = acts[s]
        for u in [f] + list(hands):
            if u and u[0] == "PICKUP" and len(u) >= 2:
                need[u[1]] = need.get(u[1], 0) + (int(u[2]) if len(u) >= 3 else 1)
    return need


def _te_add_sell(market, item, q, pos):
    """Add/merge a SELL without evicting any base order from the 10 processed slots."""
    for o in market:
        if o and len(o) >= 3 and o[0] == 'SELL' and o[1] == item:
            o[2] = int(o[2]) + q
            return True
    if len(market) >= 10:
        holes = [i for i, o in enumerate(market) if not o]
        if not holes:
            return False
        del market[holes[-1]]
    if pos == 'front':
        market.insert(0, ['SELL', item, q])
    else:
        market.append(['SELL', item, q])
    return True


# ---- exact per-item lockstep margin (same model as the engine's _process_market) ----
def _te_lockstep(orders_me, orders_opp, inv0, stock_me, stock_opp, params):
    inv = dict(inv0); stock = [dict(stock_me), dict(stock_opp)]; rev = [0.0, 0.0]
    queues = [list(orders_me), list(orders_opp)]
    for i in range(max(len(queues[0]), len(queues[1]))):
        rem = [None, None]
        for p in (0, 1):
            if i < len(queues[p]):
                o = queues[p][i]
                if o and len(o) >= 3 and o[0] in ('SELL', 'BUY_PRODUCT') and o[1] in params:
                    n = int(o[2])
                    if n > 0: rem[p] = [o[0], o[1], n]
        guard = 0
        while guard < 5000:
            guard += 1
            quoted = [None, None]
            for p in (0, 1):
                r = rem[p]
                if r is None or r[2] <= 0: continue
                if r[0] == 'SELL':
                    quoted[p] = ('SELL', r[1], _te_price(r[1], inv[r[1]], params))
                elif r[1] in ('WHEAT', 'FERTILIZER'):
                    quoted[p] = ('BUY_PRODUCT', r[1], _te_price(r[1], inv[r[1]] - 1, params))
                else:
                    rem[p] = None
            if quoted[0] is None and quoted[1] is None: break
            committed = False
            for p in (0, 1):
                q = quoted[p]
                if q is None: continue
                op, item, price = q
                if op == 'SELL':
                    if stock[p].get(item, 0) <= 0:
                        rem[p] = None; continue
                    stock[p][item] -= 1; rev[p] += price
                    if price > 1: inv[item] += 1
                else:
                    stock[p][item] = stock[p].get(item, 0) + 1; rev[p] -= price; inv[item] -= 1
                rem[p][2] -= 1; committed = True
            if not committed: break
    return rev[0], rev[1]


def _te_margin(opp, inv0, stock, params):
    cache = {}
    opp_sched = {}
    for i, order in enumerate(opp):
        if order and len(order) >= 3 and order[0] in ('SELL', 'BUY_PRODUCT') and order[1] in params:
            opp_sched.setdefault(order[1], [[] for _ in opp])[i] = order

    def margin(cand):
        sched = {}
        for i, order in enumerate(cand):
            if order and len(order) >= 3 and order[0] in ('SELL', 'BUY_PRODUCT') and order[1] in params:
                sched.setdefault(order[1], []).append((i, order[0], int(order[2])))
        total = 0.0
        for item, s in sched.items():
            key = (item, tuple(s))
            v = cache.get(key)
            if v is None:
                mine = [[] for _ in cand]
                for i, op, n in s: mine[i] = [op, item, n]
                theirs = opp_sched.get(item, [[] for _ in opp])
                a, b = _te_lockstep(mine, theirs, {item: inv0[item]}, {item: stock.get(item, 0)},
                                    {item: stock.get(item, 0)}, {item: params[item]})
                v = cache[key] = a - b
            total += v
        return total
    return margin


def _te_level2(obs, market, stock, models):
    bought = {o[1] for o in market if o and len(o) > 1 and o[0] == 'BUY_PRODUCT'}
    sells = [(i, o) for i, o in enumerate(market) if o and o[0] == 'SELL' and o[1] not in bought]
    if len(sells) < 2 or len(sells) > 6:
        return market
    slots = [i for i, _ in sells]
    orders = [o for _, o in sells]
    params = _te_params(obs)
    inv0 = {k: int(v) for k, v in obs['market']['inventory'].items()}
    fns = [_te_margin(m if m is not None else market, inv0, stock, params) for m in models]

    def score(c):
        return min(f(c) for f in fns)
    best = base = score(market); best_m = None
    for perm in _te_it.permutations(range(len(orders))):
        cand = list(market)
        for s, p in zip(slots, perm):
            cand[s] = orders[p]
        if cand == market: continue
        v = score(cand)
        if v > best + 0.5:
            best, best_m = v, cand
    if best_m is None:
        return market
    _TE_REPORT['l2_turns'] += 1
    return best_m


def _te_track(obs, st):
    prev = st.get('prev')
    step = int(obs['step'])
    if not prev or prev['step'] != step - 1:
        return
    inv = obs['market']['inventory']
    drain = _te_drain(prev['step'], prev['shops'])
    for item in _TE_PREM:
        opp = int(inv[item]) - prev['inv'][item] + drain.get(item, 0) - prev['added'].get(item, 0)
        if opp > 0 and prev['held'].get(item, 0) > 0 and item not in prev['base_sells']:
            st.setdefault('pre_steps', []).append(prev['step'])


def _te_remember(obs, st, base_market, market, stock):
    params = _te_params(obs)
    inv = {k: int(v) for k, v in obs['market']['inventory'].items()}
    left = dict(stock); added = {}
    for o in market:
        if o and len(o) >= 3 and o[0] == 'SELL' and o[1] in params:
            item = o[1]
            n = min(max(0, int(o[2])), left.get(item, 0))
            for _ in range(n):
                if _te_price(item, inv[item], params) > 1:
                    inv[item] += 1; added[item] = added.get(item, 0) + 1
            left[item] = left.get(item, 0) - n
    st['prev'] = dict(step=int(obs['step']), inv={k: int(v) for k, v in obs['market']['inventory'].items()},
                      shops=list((obs.get('town') or {}).get('unlocked_shops') or []), added=added,
                      held={k: v for k, v in left.items() if v > 0},
                      base_sells={o[1] for o in base_market if o and len(o) >= 3 and o[0] == 'SELL' and int(o[2]) > 0})


def _te_apply(obs, action, st):
    cfg = _TE_CFG
    step = int(obs['step'])
    market = [list(o) if isinstance(o, (list, tuple)) else o for o in (action.get('market') or [])]
    base_market = [list(o) for o in market]
    stock = _te_proj(obs, action)
    params = _te_params(obs)
    inv = {k: int(v) for k, v in obs['market']['inventory'].items()}
    sold = _te_sold(market)

    # (b) floor guard: trim tape SELL units that would fill at <= fg_px before fg_until
    if cfg.get('fg') and step < cfg['fg_until']:
        shed_total = sum(v for v in stock.values() if v > 0)
        will_sell = sum(min(q, stock.get(k, 0)) for k, q in sold.items())
        kept = 0
        cur = dict(inv)
        for o in market:
            if not (o and len(o) >= 3 and o[0] == 'SELL' and o[1] in cfg['fg_items']):
                continue
            item = o[1]
            n = min(max(0, int(o[2])), stock.get(item, 0))
            q = 0
            while q < n and _te_price(item, cur[item] + q, params) > cfg['fg_px']:
                q += 1
            keep = n - q
            if keep > 0 and shed_total - will_sell + kept + keep <= cfg['fg_shed']:
                o[2] = q; kept += keep; _TE_REPORT['fg_units'] += keep
            cur[item] += q
        market = [o if not (o and o[0] == 'SELL' and int(o[2]) <= 0) else [] for o in market]
        sold = _te_sold(market)

    # (d) sell-ahead of held premium stock (+ armed full sell-ahead in crashed markets)
    if cfg.get('sa') and cfg['sa_from'] <= step < cfg['liq_from']:
        reserve = _te_future_pickups(step + 1, cfg['reserve_h'])
        if cfg.get('arm_k') is not None and not st.get('armed'):
            if sum(1 for s in st.get('pre_steps', []) if s < cfg['arm_before']) >= cfg['arm_k']:
                st['armed'] = True; _TE_REPORT['armed_at'] = step
        for item in cfg['sa_items']:
            rem = stock.get(item, 0) - sold.get(item, 0) - reserve.get(item, 0)
            if rem <= 0:
                continue
            pb = params[item]['base']
            qmax = cfg['sa_max']
            armed = cfg.get('arm_k') is None or st.get('armed')
            if cfg.get('full_below') is not None and armed and _te_price(item, inv[item], params) < cfg['full_below'] * pb:
                qmax = 100
            q = 0
            cur = inv[item] + sold.get(item, 0)
            while q < rem and q < qmax and _te_price(item, cur + q, params) > cfg['sa_min_abs']:
                q += 1
            if q > 0 and _te_add_sell(market, item, q, cfg['sa_pos']):
                _TE_REPORT['sa_units'] += q
        sold = _te_sold(market)

    # (c) end-game liquidation of any leftover shed stock
    if cfg.get('liq') and step >= cfg['liq_from']:
        for item in _TE_ALL:
            rem = stock.get(item, 0) - sold.get(item, 0)
            if step < 718:
                rem -= _te_future_pickups(step + 1, 2).get(item, 0)
            if rem > 0 and _te_add_sell(market, item, rem, 'end'):
                _TE_REPORT['liq_units'] += rem
        sold = _te_sold(market)

    # (a) sells to the front: move SELLs ahead of HIRE/BUY/BUY_LAND (relative order kept; round-trip items stay)
    if cfg.get('front'):
        bought = {o[1] for o in market if o and len(o) > 1 and o[0] == 'BUY_PRODUCT'}
        sells = [o for o in market if o and o[0] == 'SELL' and o[1] not in bought]
        rest = [o for o in market if not (o and o[0] == 'SELL' and o[1] not in bought)]
        new = sells + rest
        if new != market:
            _TE_REPORT['front_turns'] += 1
            market = new

    # L2 robust ordering of our SELLs
    if cfg.get('l2'):
        market = _te_level2(obs, market, stock, [base_market] if cfg.get('l2_model') == 'base' else [base_market, None])

    try:
        _te_remember(obs, st, base_market, market, stock)
    except Exception:
        _TE_REPORT['errors'] += 1
    return dict(action, market=market[:10])


def tape_edge_agent(observation, configuration=None):
    action = _TE_BASE(observation, configuration)
    try:
        step = int(observation['step']); seat = int(observation['player'])
        st = _TE_STATE.get(seat)
        if st is None or step <= st.get('step', -1):
            st = _TE_STATE[seat] = {'step': -1}
            if step == 0:
                for k in _TE_REPORT: _TE_REPORT[k] = 0
                _TE_REPORT['armed_at'] = -1
        st['step'] = step
        if not isinstance(action, dict):
            return action
        try:
            _te_track(observation, st)
        except Exception:
            _TE_REPORT['errors'] += 1
        return _te_apply(observation, action, st)
    except Exception:
        _TE_REPORT['errors'] += 1
        return action


tape_edge_agent.telemetry = _TE_REPORT
