"""Per-day farm-layout signatures of tapes, for a day-boundary tape router.

Re-simulates each tape in its recorded world (seed + forced shops/weeds, opponent = the recorded opponent
tape) and stores, for every day start d*24, a hash of the farm layout (tile kind + crop/animal per cell +
unlocked quadrants), plus the shop list visible at that step.

    python -m harness.tape_sigs --teams DSM "Vadim Vasilenko" ... --out runs/tape_sigs.json
"""
import argparse
import glob
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from harness.fastsim import play  # noqa: E402


def layout_str(farm):
    """100-char layout: one char per cell (kind+content code); '#' locked."""
    code = {"WHEAT": "w", "CARROT": "c", "TOMATO": "t", "STRAWBERRY": "s", "MELON": "m", "COW": "C", "SHEEP": "S", "GOOSE": "G"}
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
                out.append(code.get(t.get("crop"), "?"))
            elif t.get("animal"):
                out.append(code.get(t.get("animal"), "?"))
            else:
                out.append("P" if t.get("kind") == "PASTURE" else "O")
    return "".join(out)


def layout_sig(farm):
    parts = []
    for y, row in enumerate(farm["tiles"]):
        for x, t in enumerate(row):
            if isinstance(t, dict):
                parts.append(f"{x}{y}{t.get('kind', '')[:2]}{(t.get('crop') or t.get('animal') or '')[:2]}")
    parts.append("|" + ",".join(sorted(farm.get("unlocked_quadrants") or [])))
    return hashlib.md5(";".join(parts).encode()).hexdigest()[:12]


def tape_sigs(tdir):
    t = json.load(open(os.path.join(tdir, "tape.json")))
    me = t["player"]
    opp = tdir[:-1] + str(1 - me)
    sigs, shops, money, lays = {}, {}, {}, {}

    acts = t["actions"]

    def spy(obs, cfg=None):
        s = obs["step"]
        if s % 24 == 0:
            f = obs["farms"][me]
            sigs[s // 24] = layout_sig(f)
            lays[s // 24] = layout_str(f)
            shops[s // 24] = list(obs["town"]["unlocked_shops"])
            money[s // 24] = int(f["money"])
        a = acts[s] if s < len(acts) and acts[s] else None
        return a or {"farmer": ["PASS"], "hands": [], "market": []}

    seats = [None, None]
    seats[me] = spy
    seats[1 - me] = os.path.join(opp, "main.py") if os.path.exists(os.path.join(opp, "main.py")) else "pass"
    if seats[1 - me] == "pass":
        seats[1 - me] = lambda o, c=None: {"farmer": ["PASS"], "hands": [], "market": []}
    forced = {"seat": me, "weeds": t["weeds_by_day"], "shops": t["shops_by_day"]}
    r = play(seats, seed=t["seed"], forced=forced)
    return {"team": t["team"], "episode": t["episode"], "player": me, "sigs": sigs, "shops": shops,
            "money": money, "lay": lays, "final": r["money"][me], "orig": (t.get("rewards") or [0, 0])[me]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--teams", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = {}
    if os.path.exists(a.out):
        res = json.load(open(a.out))
    for d in sorted(glob.glob("tapes/*")):
        p = os.path.join(d, "tape.json")
        if not os.path.exists(p) or os.path.basename(d) in res:
            continue
        t = json.load(open(p))
        if t["team"] not in a.teams or "shops_by_day" not in t:
            continue
        try:
            res[os.path.basename(d)] = tape_sigs(d)
        except Exception as e:
            print("fail", d, e)
    json.dump(res, open(a.out, "w"))
    print(len(res), "tapes")
