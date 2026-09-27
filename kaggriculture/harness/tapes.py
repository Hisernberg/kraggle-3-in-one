"""Turn Kaggle replays into 'tape' opponents: an agent that re-issues a real player's recorded
actions turn by turn. Farm actions only touch the player's own farm and sell/buy counts depend
only on its own shed, so on the replay's seed a tape reproduces that player's production exactly
and its market timing approximately (it cannot react to prices we change).

    python -m harness.tapes build replays/top/*.json --out tapes/      # writes tapes/<ep>_<p>/{main.py,tape.json}
    python -m harness.tapes verify tapes/                               # tape-vs-tape must reproduce rewards
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TAPE_AGENT = '''import json, os
_T = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "tape.json")))
_A = _T["actions"]
def agent(obs, config=None):
    s = obs["step"] if "step" in obs else obs["day"] * 24 + obs["hour"]
    a = _A[s] if s < len(_A) else None
    return a if a else {"farmer": ["PASS"], "hands": [], "market": []}
'''


def build(paths, out):
    made = []
    for p in paths:
        d = json.load(open(p))
        ep = d["info"].get("EpisodeId")
        seed = d["info"].get("seed")
        names = d["info"].get("TeamNames") or ["p0", "p1"]
        steps = d["steps"]
        for i in range(2):
            acts = [steps[t + 1][i].get("action") for t in range(len(steps) - 1)]
            dd = os.path.join(out, f"{ep}_{i}")
            os.makedirs(dd, exist_ok=True)
            json.dump({"episode": ep, "seed": seed, "player": i, "team": names[i], "opponent": names[1 - i],
                       "rewards": d.get("rewards"), "actions": acts}, open(os.path.join(dd, "tape.json"), "w"))
            open(os.path.join(dd, "main.py"), "w").write(TAPE_AGENT)
            made.append(dd)
    return made


def verify(tdir):
    from harness.fastsim import play
    bad = 0
    eps = {}
    for dd in sorted(glob.glob(os.path.join(tdir, "*_0"))):
        t = json.load(open(os.path.join(dd, "tape.json")))
        other = dd[:-2] + "_1"
        r = play([os.path.join(dd, "main.py"), os.path.join(other, "main.py")], seed=t["seed"])
        ok = [round(x) for x in r["money"]] == [round(x) for x in (t["rewards"] or [0, 0])]
        bad += not ok
        eps[t["episode"]] = ok
        if not ok:
            print("MISMATCH", dd, r["money"], t["rewards"])
    print(f"{len(eps)} episodes, {bad} mismatches")


if __name__ == "__main__":
    if sys.argv[1] == "build":
        args = sys.argv[2:]
        out = args[args.index("--out") + 1]
        files = [a for a in args if a.endswith(".json")]
        print(len(build(files, out)), "tapes")
    else:
        verify(sys.argv[2])
