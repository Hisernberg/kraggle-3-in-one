"""Round-robin / gauntlet tournaments on the fast simulator, with Bradley-Terry ratings.

    python -m harness.tournament --agents a/main.py b/main.py c/main.py --seeds 4 --workers 3 --out runs/rr1.jsonl
    python -m harness.tournament --hero agents/x/main.py --field-file field.txt --seeds 6 --out runs/g.jsonl

Every pairing is played on each seed in both seat orders (seat 0 wins exact market-order ties).
Results append to a JSONL file so interrupted runs resume (already-played games are skipped).
"""
import argparse
import itertools
import json
import math
import multiprocessing as mp
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from harness.fastsim import play  # noqa: E402


def _name(p):
    p = os.path.normpath(p)
    return os.path.basename(os.path.dirname(p)) if os.path.basename(p) in ("main.py", "submission.py") else p


def _job(args):
    a, b, seed = args
    try:
        r = play([a, b], seed=seed)
    except Exception as e:  # loader failure etc.
        return {"a": a, "b": b, "seed": seed, "crash": repr(e)}
    return {"a": a, "b": b, "seed": seed, "money": r["money"], "status": r["status"],
            "time": r["time"], "max_turn_time": r["max_turn_time"], "errors": r["errors"], "shops": r["shops"]}


def outcome(rec):
    """Score for seat a: 1 win, 0.5 tie, 0 loss (errors lose)."""
    if "crash" in rec:
        return None
    sa, sb = rec["status"]
    if sa != "ACTIVE" and sb != "ACTIVE":
        return 0.5
    if sa != "ACTIVE":
        return 0.0
    if sb != "ACTIVE":
        return 1.0
    ma, mb = rec["money"]
    return 1.0 if ma > mb else 0.0 if ma < mb else 0.5


def bradley_terry(games, iters=200):
    """games: list of (i, j, score_i). Returns log-strength*400/ln10 (Elo-like), mean 0."""
    players = sorted({g[0] for g in games} | {g[1] for g in games})
    w = defaultdict(float)
    n = defaultdict(float)
    for i, j, s in games:
        w[i] += s
        w[j] += 1 - s
        n[(i, j)] += 1
        n[(j, i)] += 1
    p = {x: 1.0 for x in players}
    for _ in range(iters):
        newp = {}
        for i in players:
            den = sum(n[(i, j)] / (p[i] + p[j]) for j in players if j != i and n[(i, j)])
            newp[i] = (w[i] + 0.5) / (den + 1.0 / (p[i] + 1.0)) if den else p[i]  # weak prior
        g = math.exp(sum(math.log(v) for v in newp.values()) / len(newp))
        p = {k: v / g for k, v in newp.items()}
    return {k: 400 * math.log10(v) for k, v in p.items()}


def summarize(recs, hero=None):
    stats = defaultdict(lambda: {"g": 0, "w": 0.0, "margin": 0.0, "money": 0.0, "err": 0, "tmax": 0.0})
    games = []
    h2h = defaultdict(lambda: [0.0, 0])
    for r in recs:
        s = outcome(r)
        if s is None:
            continue
        a, b = _name(r["a"]), _name(r["b"])
        games.append((a, b, s))
        for who, opp, sc, k in ((a, b, s, 0), (b, a, 1 - s, 1)):
            st = stats[who]
            st["g"] += 1
            st["w"] += sc
            st["money"] += r["money"][k]
            st["margin"] += r["money"][k] - r["money"][1 - k]
            st["err"] += r["status"][k] != "ACTIVE"
            st["tmax"] = max(st["tmax"], r["max_turn_time"][k])
            h2h[(who, opp)][0] += sc
            h2h[(who, opp)][1] += 1
    bt = bradley_terry(games) if games else {}
    rows = sorted(stats, key=lambda k: -bt.get(k, 0))
    lines = [f"{'agent':58s} {'BT':>6s} {'games':>5s} {'win%':>6s} {'avg$':>8s} {'margin':>8s} {'err':>3s} {'tmax':>5s}"]
    for k in rows:
        st = stats[k]
        lines.append(f"{k[:58]:58s} {bt.get(k, 0):6.0f} {st['g']:5d} {100 * st['w'] / st['g']:6.1f} "
                     f"{st['money'] / st['g']:8.0f} {st['margin'] / st['g']:8.0f} {st['err']:3d} {st['tmax']:5.2f}")
    if hero:
        hn = _name(hero)
        lines.append(f"\nhead-to-head for {hn}:")
        for (x, y), (sc, g) in sorted(h2h.items(), key=lambda kv: kv[1][0] / kv[1][1]):
            if x == hn:
                lines.append(f"  vs {y[:56]:56s} {sc:4.1f}/{g}")
    return "\n".join(lines), bt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agents", nargs="*", default=[])
    ap.add_argument("--field-file")
    ap.add_argument("--hero", nargs="*", default=[], help="only play hero(es) vs field (gauntlet)")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--seed0", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--out", required=True)
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args()

    field = list(args.agents)
    if args.field_file:
        field += [l.strip() for l in open(args.field_file) if l.strip() and not l.startswith("#")]
    seeds = [args.seed0 + k for k in range(args.seeds)]
    if args.hero:
        pairs = [(h, f) for h in args.hero for f in field if f != h]
    else:
        pairs = list(itertools.combinations(field, 2))
    jobs = [(a, b, s) for (x, y) in pairs for s in seeds for (a, b) in ((x, y), (y, x))]

    done = []
    if os.path.exists(args.out):
        done = [json.loads(l) for l in open(args.out) if l.strip()]
    have = {(r["a"], r["b"], r["seed"]) for r in done}
    todo = [j for j in jobs if j not in have]
    if not args.summary_only and todo:
        print(f"{len(todo)} games to play ({len(jobs) - len(todo)} cached)", flush=True)
        t0 = time.time()
        ctx = mp.get_context("fork")
        with ctx.Pool(args.workers, maxtasksperchild=1) as pool, open(args.out, "a") as f:
            for k, rec in enumerate(pool.imap_unordered(_job, todo), 1):
                f.write(json.dumps(rec) + "\n")
                f.flush()
                done.append(rec)
                if "crash" in rec or any(e for e in rec.get("errors", [])):
                    print("PROBLEM:", _name(rec["a"]), _name(rec["b"]), rec.get("crash") or rec["errors"], flush=True)
                if k % 20 == 0:
                    print(f"  {k}/{len(todo)} games, {time.time() - t0:.0f}s", flush=True)
    wanted = set(jobs)
    recs = [r for r in done if (r["a"], r["b"], r["seed"]) in wanted]
    text, _ = summarize(recs, hero=args.hero[0] if args.hero else None)
    print(text)


if __name__ == "__main__":
    main()
