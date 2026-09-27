"""Daily UMUD submission loop (5 Kaggle submissions per UTC day).

State lives in daily/state.json (best blend args + score + experiment queue) and every submission is
appended to daily/history.csv, so a fresh container can continue from the repo alone (plus Kaggle
credentials in ~/.kaggle/kaggle.json).

    python daily/run.py status                      # quota, leaderboard top 6, current best, queue
    python daily/run.py build NAME [extra blend args]   # base (best) args + overrides -> daily/out/NAME.csv
    python daily/run.py submit NAME "message"       # submit daily/out/NAME.csv, wait, log, update best
"""
import csv
import datetime as dt
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

D = Path(__file__).resolve().parent
UMUD = D.parent
STATE = D / "state.json"
HIST = D / "history.csv"
OUT = D / "out"
COMP = "umud-challenge-muscle-architecture-in-ultrasound-data"
INPUTS = ["--ref", "daily/inputs/ref_vera_public.csv", "--groups", "daily/inputs/test_groups.npy",
          "--pipeline", "daily/inputs/pipeline_hyb.csv", "--features", "daily/inputs/features_hyb.csv"]


def sh(cmd):
    return subprocess.run(cmd, capture_output=True, text=True, cwd=UMUD).stdout


def load():
    return json.loads(STATE.read_text())


def merge_args(base, extra):
    """Override base flags with extra flags (flag followed by its values)."""
    def parse(tokens):
        out, cur = {}, None
        for t in tokens:
            if t.startswith("--"):
                cur = t
                out[cur] = []
            else:
                out[cur].append(t)
        return out
    b, e = parse(base), parse(extra)
    b.update(e)
    return [x for k, v in b.items() for x in [k, *v]]


def status():
    s = load()
    print(sh(["kaggle", "competitions", "submission-limits", COMP]).strip().splitlines()[-1])
    lb = sh(["kaggle", "competitions", "leaderboard", COMP, "-s"]).splitlines()[3:10]
    print("leaderboard:", " | ".join(" ".join(l.split()[1:2] + l.split()[-1:]) for l in lb if l.strip()))
    print("best:", s["best_score"], s["best_name"], " ".join(s["best_args"]))
    print("queue:")
    for q in s["queue"]:
        print("  -", q)


def build(name, extra):
    s = load()
    OUT.mkdir(exist_ok=True)
    args = merge_args(s["best_args"], extra)
    cmd = [sys.executable, "scripts/blend_v2.py", *INPUTS, *args, "--out", str(OUT / f"{name}.csv")]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=UMUD)
    print(r.stdout.strip() or r.stderr.strip())
    (OUT / f"{name}.args").write_text(" ".join(args))


def submit(name, msg):
    f = OUT / f"{name}.csv"
    subprocess.run(["kaggle", "competitions", "submit", COMP, "-f", str(f), "-m", msg], capture_output=True, cwd=UMUD)
    while True:
        rows = sh(["kaggle", "competitions", "submissions", COMP]).splitlines()
        top = rows[3] if len(rows) > 3 else ""
        if f.name in top and "PENDING" not in top:
            break
        time.sleep(10)
    score = float(top.split()[-1])
    args = (OUT / f"{name}.args").read_text() if (OUT / f"{name}.args").exists() else ""
    s = load()
    new_best = score < s["best_score"]
    with HIST.open("a", newline="") as fh:
        csv.writer(fh).writerow([dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M"), name, score, int(new_best), args, msg])
    if new_best:
        s.update(best_score=score, best_name=name, best_args=shlex.split(args))
        STATE.write_text(json.dumps(s, indent=1) + "\n")
    print(f"{name} {score} {'NEW BEST' if new_best else '(best ' + str(s['best_score']) + ')'}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd == "status":
        status()
    elif cmd == "build":
        build(sys.argv[2], sys.argv[3:])
    elif cmd == "submit":
        submit(sys.argv[2], sys.argv[3])
