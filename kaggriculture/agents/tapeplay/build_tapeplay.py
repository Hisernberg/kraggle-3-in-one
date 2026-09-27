"""Build a 'tapeplay' agent: a top player's recorded farm actions and purchases, with adaptive selling.

    python agents/tapeplay/build_tapeplay.py tapes/113957215_0 [name] [--switch-day 8]

The recorded unit actions (farmer + hands) and all non-SELL market orders (hires, seeds, animals, land,
wheat/fertilizer buys) are replayed verbatim. Recorded SELL orders are kept until the switch day (they
fund the cash-tight opening); afterwards selling is decided by sell_policy.py against the live market.
"""
import json
import os
import sys

here = os.path.dirname(os.path.abspath(__file__))
tdir = sys.argv[1]
name = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else os.path.basename(tdir.rstrip("/"))
switch = 8
if "--switch-day" in sys.argv:
    switch = int(sys.argv[sys.argv.index("--switch-day") + 1])
t = json.load(open(os.path.join(tdir, "tape.json")))
acts = []
for a in t["actions"]:
    a = a or {}
    acts.append([a.get("farmer") or ["PASS"], a.get("hands") or [], a.get("market") or []])
policy = open(os.path.join(here, "sell_policy.py")).read()
src = f'''# tapeplay agent built from episode {t["episode"]} player {t["player"]} ({t["team"]}); sells adaptive from day {switch}
import json as _tp_json
_TP_ACTS = _tp_json.loads({json.dumps(json.dumps(acts, separators=(",", ":")))})
_TP_SWITCH_DAY = {switch}
''' + policy
d = os.path.join(here, name)
os.makedirs(d, exist_ok=True)
open(os.path.join(d, "main.py"), "w").write(src)
print(os.path.join(d, "main.py"), len(src) // 1024, "KB")
