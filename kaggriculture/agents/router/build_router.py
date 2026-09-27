"""Build a day-boundary tape router.

    python agents/router/build_router.py NAME [--teams DSM "Vadim Vasilenko" ...] [--max-switch-day 8]

At every day start (units respawn at the shed, inventories are empty) the router looks for recorded
games whose farm layout at that day start equals ours, and switches to the one whose town-shop draws
so far best match our game (plus a robustness prior from local screening). Between day starts it replays
the chosen tape verbatim (with land catch-up, as in tapeplay).
"""
import base64
import glob
import gzip
import json
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
name = sys.argv[1]
teams = None
if "--teams" in sys.argv:
    i = sys.argv.index("--teams") + 1
    teams = []
    while i < len(sys.argv) and not sys.argv[i].startswith("--"):
        teams.append(sys.argv[i])
        i += 1
max_day = int(sys.argv[sys.argv.index("--max-switch-day") + 1]) if "--max-switch-day" in sys.argv else 8
exclude_seeds = set()

sigs = json.load(open(os.path.join(ROOT, "runs/tape_sigs.json")))
# robustness prior: win rate of raw_<tape> in every local screening file
wins, games = defaultdict(float), defaultdict(int)
for f in glob.glob(os.path.join(ROOT, "runs/tapescreen*.jsonl")) + glob.glob(os.path.join(ROOT, "runs/tapeval.jsonl")):
    for line in open(f):
        r = json.loads(line)
        if "money" not in r:
            continue
        for k, who in enumerate((r["a"], r["b"])):
            if "/raw_" in who:
                tid = who.split("/raw_")[1].split("/")[0]
                games[tid] += 1
                wins[tid] += r["money"][k] > r["money"][1 - k]
data = {}
for tid, v in sigs.items():
    if teams and v["team"] not in teams:
        continue
    t = json.load(open(os.path.join(ROOT, "tapes", tid, "tape.json")))
    acts = [[(a or {}).get("farmer") or ["PASS"], (a or {}).get("hands") or [], (a or {}).get("market") or []]
            for a in t["actions"]]
    n = games.get(tid, 0)
    prior = (wins.get(tid, 0) + 1.0) / (n + 2.0)          # Laplace-smoothed win rate
    won_orig = 1.0 if v["orig"] >= max((t.get("rewards") or [0, 0])) else 0.0
    data[tid] = {"a": acts, "l": {d: v["lay"][d] for d in v["lay"] if int(d) <= max_day},
                 "sh": {d: v["shops"][d] for d in v["shops"] if int(d) <= max_day},
                 "p": round(prior, 3), "n": n, "w": won_orig}
blob = base64.b64encode(gzip.compress(json.dumps(data, separators=(",", ":")).encode(), 9)).decode()
core = open(os.path.join(HERE, "router_core.py")).read()
src = f"# tape router: {len(data)} tapes, switch days <= {max_day}\n_RT_BLOB = {blob!r}\n_RT_MAX_DAY = {max_day}\n" + core
d = os.path.join(HERE, name)
os.makedirs(d, exist_ok=True)
open(os.path.join(d, "main.py"), "w").write(src)
print(os.path.join(d, "main.py"), len(data), "tapes", len(src) // 1024, "KB")
