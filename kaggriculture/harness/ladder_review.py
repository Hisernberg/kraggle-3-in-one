"""Ladder review for one submission: rating, W/L, margins, opponent classes, day-1 cash check.

    python -m harness.ladder_review SUBMISSION_ID [--n 40] [--dir runs/ladder]

Downloads the newest N public episodes of the submission (cached), then prints one line per game and a summary
by opponent turn-0 opening class ("duel" = BUY wheat then SELL wheat on turn 0; "animal-first" = template-A
style; "buy5" = cha22 family). The team name is TEAM below.
"""
import argparse
import collections
import glob
import json
import os
import subprocess

TEAM = "Fried Chicken Lovers"


def fetch(sub, n, out):
    os.makedirs(out, exist_ok=True)
    csv = subprocess.run(["kaggle", "competitions", "episodes", str(sub), "--csv"], capture_output=True, text=True).stdout
    eps = [ln.split(",")[0] for ln in csv.splitlines()[1:] if "PUBLIC" in ln][:n]
    for ep in eps:
        if not os.path.exists(os.path.join(out, f"episode-{ep}-replay.json")):
            subprocess.run(["kaggle", "competitions", "replay", ep, "-p", out], capture_output=True)
    return eps


def opening_class(market):
    k = [o[0] for o in market[:2]]
    if k and k[0] == "BUY_ANIMAL":
        return "animal-first"
    if len(k) > 1 and k[0] == "BUY_PRODUCT" and k[1] == "SELL":
        return "duel"
    if k and k[0] == "BUY_PRODUCT" and market[0][1:3] == ["WHEAT", 5]:
        return "buy5"
    return "other"


def review(paths):
    rows = []
    for f in paths:
        try:
            d = json.load(open(f))
        except Exception:
            continue
        names = d["info"]["TeamNames"]
        if TEAM not in names:
            continue
        me = names.index(TEAM)
        op = 1 - me
        st = d["steps"]
        rw = d["rewards"]
        cash24 = int(st[24][0]["observation"]["farms"][me]["money"]) if len(st) > 24 else -1
        hands25 = len(st[25][0]["observation"]["farms"][me]["hands"]) if len(st) > 25 else -1
        cls = opening_class(st[1][op]["action"].get("market", []))
        rows.append({"ep": os.path.basename(f)[8:17], "win": rw[me] > rw[op], "me": int(rw[me]), "opp": int(rw[op]),
                     "margin": int(rw[me] - rw[op]), "opp_name": names[op], "cls": cls, "cash24": cash24,
                     "hands25": hands25})
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("sub")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--dir", default="runs/ladder")
    a = ap.parse_args()
    eps = fetch(a.sub, a.n, a.dir)
    rows = review([os.path.join(a.dir, f"episode-{e}-replay.json") for e in eps])
    for r in sorted(rows, key=lambda r: r["margin"]):
        print(f"{r['ep']} {'W' if r['win'] else 'L'} {r['margin']:+7d} me={r['me']:6d} opp={r['opp']:6d} "
              f"{r['cls']:12s} cash24={r['cash24']:4d} hands25={r['hands25']} {r['opp_name'][:20]}")
    by = collections.defaultdict(list)
    for r in rows:
        by[r["cls"]].append(r)
    print(f"\n{len(rows)} games, {sum(r['win'] for r in rows)} wins")
    for c, rs in sorted(by.items(), key=lambda x: -len(x[1])):
        print(f"  {c:12s} n={len(rs):3d} win={sum(r['win'] for r in rs) / len(rs):.0%} "
              f"avg_margin={sum(r['margin'] for r in rs) / len(rs):+.0f}")
    losses = [r for r in rows if not r["win"]]
    print(f"  losses <500: {sum(abs(r['margin']) < 500 for r in losses)}  >5000: {sum(abs(r['margin']) > 5000 for r in losses)}"
          f"  day-1 cash<3: {sum(r['cash24'] < 3 for r in rows)}")
