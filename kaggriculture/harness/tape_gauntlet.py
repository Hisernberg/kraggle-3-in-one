"""Play hero agents against replay tapes of real (top) ladder players, each on its own episode seed
and in the seat the tape was recorded in.

    python -m harness.tape_gauntlet --heroes a/main.py b/main.py --tapes tapes --workers 3 --out runs/tg.jsonl
"""
import argparse
import glob
import json
import multiprocessing as mp
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from harness.fastsim import play  # noqa: E402


def _name(p):
    return os.path.basename(os.path.dirname(os.path.normpath(p)))


FORCE = True


def _job(args):
    hero, tdir = args
    t = json.load(open(os.path.join(tdir, "tape.json")))
    i = t["player"]
    seats = [None, None]
    seats[i] = os.path.join(tdir, "main.py")
    seats[1 - i] = hero
    try:
        forced = None
        if FORCE and "shops_by_day" in t:
            forced = {"seat": i, "weeds": t["weeds_by_day"], "shops": t["shops_by_day"]}
        r = play(seats, seed=t["seed"], forced=forced)
    except Exception as e:
        return {"hero": hero, "tape": tdir, "crash": repr(e)}
    return {"hero": hero, "tape": tdir, "team": t["team"], "tape_seat": i, "hero_money": r["money"][1 - i],
            "tape_money": r["money"][i], "hero_status": r["status"][1 - i], "err": r["errors"][1 - i],
            "orig": t["rewards"], "tmax": r["max_turn_time"][1 - i]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heroes", nargs="+", required=True)
    ap.add_argument("--tapes", default="tapes")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-force", action="store_true", help="do not force recorded shops/weeds")
    args = ap.parse_args()
    global FORCE
    FORCE = not args.no_force
    tdirs = sorted(d for d in glob.glob(os.path.join(args.tapes, "*")) if os.path.exists(os.path.join(d, "tape.json")))
    jobs = [(h, t) for h in args.heroes for t in tdirs]
    done = [json.loads(l) for l in open(args.out)] if os.path.exists(args.out) else []
    have = {(r["hero"], r["tape"]) for r in done}
    todo = [j for j in jobs if j not in have]
    if todo:
        print(f"{len(todo)} games", flush=True)
        t0 = time.time()
        with mp.get_context("fork").Pool(args.workers, maxtasksperchild=1) as pool, open(args.out, "a") as f:
            for k, rec in enumerate(pool.imap_unordered(_job, todo), 1):
                f.write(json.dumps(rec) + "\n")
                f.flush()
                done.append(rec)
                if k % 50 == 0:
                    print(f"  {k}/{len(todo)} {time.time() - t0:.0f}s", flush=True)
    want = set(jobs)
    recs = [r for r in done if (r["hero"], r["tape"]) in want and "crash" not in r]
    by = defaultdict(list)
    for r in recs:
        by[r["hero"]].append(r)
    print(f"{'hero':50s} {'games':>5s} {'win%':>6s} {'hero$':>8s} {'tape$':>8s} {'margin':>7s} {'err':>3s} {'tmax':>5s}")
    rows = []
    for h, rs in by.items():
        w = sum((r["hero_status"] == "ACTIVE") * ((r["hero_money"] > r["tape_money"]) + 0.5 * (r["hero_money"] == r["tape_money"])) for r in rs)
        rows.append((w / len(rs), h, rs))
    for wr, h, rs in sorted(rows, reverse=True):
        n = len(rs)
        print(f"{_name(h)[:50]:50s} {n:5d} {100 * wr:6.1f} {sum(r['hero_money'] for r in rs) / n:8.0f} "
              f"{sum(r['tape_money'] for r in rs) / n:8.0f} {sum(r['hero_money'] - r['tape_money'] for r in rs) / n:7.0f} "
              f"{sum(r['hero_status'] != 'ACTIVE' for r in rs):3d} {max(r['tmax'] for r in rs):5.2f}")
    if len(by) <= 3:
        for h, rs in by.items():
            team = defaultdict(lambda: [0, 0, 0.0])
            for r in rs:
                e = team[r["team"]]
                e[0] += r["hero_money"] > r["tape_money"]
                e[1] += 1
                e[2] += r["hero_money"] - r["tape_money"]
            print(f"\n{_name(h)} by tape team:")
            for k, (w, n, m) in sorted(team.items(), key=lambda kv: kv[1][0] / kv[1][1]):
                print(f"  {k[:30]:30s} {w}/{n}  avg margin {m / n:8.0f}")


if __name__ == "__main__":
    main()
