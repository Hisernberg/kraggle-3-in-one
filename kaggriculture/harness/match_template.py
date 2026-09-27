"""Which local agent plays most like a recorded ladder player? Plays each candidate against the
recorded opponent tape on the episode seed, in the recorded player's seat, and counts how many of the
first N turns produce exactly the same action (farmer + hands + market) as the recorded player.

    python -m harness.match_template --tape tapes/113950605_1 --n 120 --agents public_agents/*/main.py
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from harness.fastsim import load_agent, play  # noqa: E402


def norm(a):
    if not a:
        return None
    return json.dumps({"f": a.get("farmer") or ["PASS"], "h": a.get("hands") or [], "m": a.get("market") or []})


def score(agent_path, tape_dir, n):
    t = json.load(open(os.path.join(tape_dir, "tape.json")))
    me = t["player"]
    opp_dir = tape_dir[:-1] + str(1 - me)
    rec = [norm(a) for a in t["actions"][:n]]
    try:
        fn = load_agent(agent_path)
    except Exception as e:
        return 0, 0, "load:" + repr(e)[:80]
    mine = []

    def spy(obs, cfg=None):
        a = fn(obs, cfg) if fn.__code__.co_argcount >= 2 else fn(obs)
        if obs["step"] < n:
            mine.append(norm(a))
        return a

    seats = [None, None]
    seats[me] = spy
    seats[1 - me] = os.path.join(opp_dir, "main.py")
    try:
        play(seats, seed=t["seed"], cfg={"episodeSteps": n + 2})
    except Exception as e:
        return 0, 0, repr(e)
    same = sum(1 for a, b in zip(mine, rec) if a == b)
    first_diff = next((i for i, (a, b) in enumerate(zip(mine, rec)) if a != b), len(mine))
    return same, first_diff, None


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tape", required=True)
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--agents", nargs="+")
    a = ap.parse_args()
    res = []
    for p in a.agents:
        s, fd, err = score(p, a.tape, a.n)
        res.append((fd, s, p, err))
    for fd, s, p, err in sorted(res, reverse=True)[:25]:
        print(f"first_diff={fd:4d} same={s:4d}/{a.n}  {os.path.basename(os.path.dirname(p))} {err or ''}")
