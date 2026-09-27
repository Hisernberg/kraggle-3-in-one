"""Opponent market-flow inference from public observations (agent-side library).

Between observation t and t+1 the market inventory changes by
    my_sold_added - my_bought + opp_sold_added - opp_bought - town_consumed(t)
Town consumption is deterministic given the shop list seen at t, and our own executed orders are
known from our shed, so the opponent's net flow per item per turn is exactly recoverable (except
sales made at the $1 floor, which the engine does not add to inventory).

Usage inside an agent:  tracker = OppTracker();  each turn: tracker.observe(obs, my_last_market_orders)
"""
from collections import defaultdict

SHOPS = {
    "BAKERY": ["EGG", "WHEAT"], "PIZZA_SHOP": ["MILK", "TOMATO", "WHEAT"],
    "BRUNCH_SPOT": ["EGG", "WHEAT", "STRAWBERRY"], "YARN_STORE": ["WOOL"],
    "ICE_CREAM_SHOP": ["STRAWBERRY", "MILK", "WHEAT"], "PET_CAFE": ["CARROT"],
    "SMOOTHIE_SHOP": ["STRAWBERRY", "MILK"], "FARMERS_MARKET": ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY"],
}
PRODUCTS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER"]


def town_consumption(step, shops, shop_interval=4, center_interval=24):
    out = defaultdict(int)
    if step % shop_interval == 0:
        for s in shops:
            prods = SHOPS.get(s, [])
            m = 2 if len(prods) == 1 else 1
            for p in prods:
                out[p] += m
    if step % center_interval == 0:
        for p in PRODUCTS:
            if p != "FERTILIZER":
                out[p] += 1
    return out


def town_rate_per_day(shops):
    """Units per day the town removes from the market for each product."""
    rate = defaultdict(float)
    for s in shops:
        prods = SHOPS.get(s, [])
        m = 2 if len(prods) == 1 else 1
        for p in prods:
            rate[p] += 6 * m
    for p in PRODUCTS:
        if p != "FERTILIZER":
            rate[p] += 1
    return rate


class OppTracker:
    def __init__(self):
        self.prev = None           # (step, inventory, shops, my_shed_total_by_item)
        self.opp_net = defaultdict(int)   # cumulative opponent net sold (sold - bought) per item
        self.opp_last_sell = {}    # item -> last step the opponent net-sold
        self.history = []          # (step, {item: opp_net_flow})

    def observe(self, obs, my_exec_flow_prev):
        """my_exec_flow_prev: {item: my net units added to market last turn (sold - bought)}."""
        step = obs["step"] if "step" in obs else obs.get("day", 0) * 24 + obs.get("hour", 0)
        inv = dict(obs["market"]["inventory"])
        shops = list(obs["town"]["unlocked_shops"])
        if self.prev is not None and self.prev[0] == step - 1:
            pstep, pinv, pshops = self.prev
            town = town_consumption(pstep, pshops)
            flow = {}
            for p in PRODUCTS:
                d = inv[p] - pinv[p] + town.get(p, 0) - my_exec_flow_prev.get(p, 0)
                if d:
                    flow[p] = d
                    self.opp_net[p] += d
                    if d > 0:
                        self.opp_last_sell[p] = pstep
            self.history.append((pstep, flow))
        self.prev = (step, inv, shops)
        return self.history[-1][1] if self.history else {}
