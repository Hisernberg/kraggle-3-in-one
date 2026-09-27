"""Aggregate replay_parse.py output (parsed/*.json) into per-team stats.

    python replay_analyze.py parsed/top.json parsed/ours.json > parsed/team_stats.txt
"""
import json
import statistics as st
import sys
from collections import Counter, defaultdict

BASE = {"WHEAT": 25, "CARROT": 35, "TOMATO": 60, "STRAWBERRY": 120, "MELON": 250, "EGG": 50, "MILK": 160,
        "WOOL": 200, "FERTILIZER": 100}


def load(paths):
    games = []
    for p in paths:
        games += json.load(open(p))
    # dedupe by path basename
    seen, out = set(), []
    for g in games:
        k = g["path"].split("/")[-1]
        if k in seen:
            continue
        seen.add(k)
        out.append(g)
    return out


def mean(xs):
    xs = list(xs)
    return round(st.mean(xs), 1) if xs else None


def team_stats(games, team):
    rows = []
    for g in games:
        for f in g["features"]:
            if f["name"] == team:
                rows.append((g, f))
    if not rows:
        return None
    S = {"team": team, "n": len(rows)}
    S["wins"] = sum(f["won"] for _, f in rows)
    S["final_mean"] = mean(f["final_money"] for _, f in rows)
    S["margin_mean"] = mean(f["final_money"] - f["opp_final"] for _, f in rows)
    S["margin_pct"] = mean(100 * (f["final_money"] - f["opp_final"]) / max(1, f["opp_final"]) for _, f in rows)
    S["opps"] = Counter(f["opp_name"] for _, f in rows)
    # animals & seeds totals
    tot = Counter()
    first_day = {}
    for _, f in rows:
        for d, c in f["buys_by_day"].items():
            for k, v in c.items():
                tot[k] += v
                first_day.setdefault(k, []).append(int(d))
    S["buys_per_game"] = {k: round(v / len(rows), 1) for k, v in tot.most_common()}
    S["first_buy_day_min"] = {k: min(v) for k, v in first_day.items()}
    # animals bought by day window
    win = defaultdict(Counter)
    for _, f in rows:
        for d, c in f["buys_by_day"].items():
            d = int(d)
            w = "d0" if d == 0 else "d1-2" if d <= 2 else "d3-5" if d <= 5 else "d6-10" if d <= 10 else "d11-20" if d <= 20 else "d21+"
            for k in ("COW", "SHEEP", "GOOSE", "STRAWBERRY", "MELON", "TOMATO", "CARROT", "WHEAT_SEED"):
                pass
            for k, v in c.items():
                win[w][k] += v
    S["buys_by_window_per_game"] = {w: {k: round(v / len(rows), 1) for k, v in c.most_common()} for w, c in
                                    sorted(win.items(), key=lambda kv: ["d0", "d1-2", "d3-5", "d6-10", "d11-20", "d21+"].index(kv[0]))}
    S["land"] = Counter(tuple((l[0], int(l[2])) for l in f["land"]) for _, f in rows).most_common(5)
    S["land_day_by_price"] = {int(p): mean(l[0] + l[1] / 24 for _, f in rows for l in f["land"] if int(l[2]) == p)
                              for p in (1000, 2000, 4000)}
    S["n_quadrants_mean"] = mean(1 + len(f["land"]) for _, f in rows)
    hires = defaultdict(list)
    for _, f in rows:
        for d in range(30):
            hires[d].append(f["hires_by_day"].get(str(d), 0))
    S["hires_by_day_mean"] = [round(st.mean(hires[d]), 1) for d in range(30)]
    S["hire_cost_mean"] = mean(f["hire_cost_total"] for _, f in rows)
    rev = Counter()
    units = Counter()
    for _, f in rows:
        for k, v in f["revenue"].items():
            rev[k] += v
        for k, v in f["units_sold"].items():
            units[k] += v
    S["revenue_per_game"] = {k: round(v / len(rows)) for k, v in rev.most_common()}
    S["units_per_game"] = {k: round(v / len(rows)) for k, v in units.most_common()}
    S["avg_price"] = {k: round(rev[k] / units[k], 1) for k in rev}
    S["price_vs_base"] = {k: round(rev[k] / units[k] / BASE[k], 2) for k in rev}
    spend = Counter()
    for _, f in rows:
        for k, v in f["spend"].items():
            spend[k] += v
    S["spend_per_game"] = {k: round(v / len(rows)) for k, v in spend.most_common()}
    # sell order sizes
    sizes = Counter()
    for _, f in rows:
        for s in f["sells"]:
            item, n = s[2], s[3]
            sizes[item + ":" + ("1" if n == 1 else "2-3" if n <= 3 else "4-9" if n <= 9 else "10+")] += 1
    S["sell_sizes"] = dict(sorted(sizes.items()))
    S["sell_orders_per_game"] = mean(f["n_sell_orders"] for _, f in rows)
    S["sells_near_opp_frac"] = mean(f["sells_near_opp_sell"] / max(1, f["n_sell_orders"]) for _, f in rows)
    # sell hour histogram for premium goods
    hh = Counter()
    for _, f in rows:
        for s in f["sells"]:
            if s[2] in ("MILK", "WOOL", "STRAWBERRY", "MELON"):
                hh[s[1]] += s[3]
    S["premium_units_by_hour"] = [hh.get(h, 0) for h in range(24)]
    # final tiles
    ft = Counter()
    for _, f in rows:
        for k, v in (f["final_tiles"] or {}).items():
            ft[k] += v
    S["final_tiles_mean"] = {k: round(v / len(rows), 1) for k, v in ft.most_common()}
    tb = defaultdict(Counter)
    for _, f in rows:
        for d, c in f["tiles_by_day"].items():
            for k, v in c.items():
                tb[int(d)][k] += v
    S["tiles_by_day_mean"] = {d: {k: round(v / len(rows), 1) for k, v in tb[d].most_common() if k not in ("LOCKED",)}
                              for d in (0, 1, 2, 3, 5, 7, 10, 15, 20, 25, 29) if d in tb}
    mb = defaultdict(list)
    for _, f in rows:
        for d, m in f["money_by_day"].items():
            mb[int(d)].append(m)
    S["money_by_day_mean"] = {d: round(st.mean(mb[d])) for d in sorted(mb) if d in (0, 1, 2, 3, 5, 7, 10, 15, 20, 25, 28, 29)}
    es = Counter()
    for _, f in rows:
        for k, v in (f["end_shed"] or {}).items():
            es[k] += v
    S["end_shed_mean"] = {k: round(v / len(rows), 1) for k, v in es.items()}
    am = Counter()
    for _, f in rows:
        for k, v in f["action_mix"].items():
            am[k] += v
    S["unit_actions_per_game"] = {k: round(v / len(rows)) for k, v in am.most_common()}
    # wheat buy product
    S["wheat_bought_units"] = mean(sum(b[3] for b in f["buy_product"] if b[2] == "WHEAT") for _, f in rows)
    S["fert_bought_units"] = mean(sum(b[3] for b in f["buy_product"] if b[2] == "FERTILIZER") for _, f in rows)
    S["t0"] = Counter(json.dumps(f["t0_wheat"]) for _, f in rows).most_common(3)
    S["day0_market_first6"] = Counter(" | ".join(l.split(" M:")[1] for l in f["opening_day0"][:3]) for _, f in rows).most_common(3)
    return S


def main():
    games = load(sys.argv[1:])
    teams = Counter(f["name"] for g in games for f in g["features"])
    order = [t for t, _ in teams.most_common()]
    out = {}
    for t in order:
        s = team_stats(games, t)
        out[t] = s
    json.dump(out, open("parsed/team_stats.json", "w"), default=str, indent=1)
    for t in order:
        s = out[t]
        if s["n"] < 3:
            continue
        print("#" * 100)
        for k, v in s.items():
            print(f"{k}: {v}")


if __name__ == "__main__":
    main()
