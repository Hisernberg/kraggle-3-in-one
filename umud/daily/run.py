"""Daily UMUD submission loop (5 Kaggle submissions per UTC day).

State lives in daily/state.json (best blend args + score + experiment queue) and every submission is
appended to daily/history.csv, so a fresh container can continue from the repo alone (plus Kaggle
credentials in ~/.kaggle/kaggle.json).

    python daily/run.py status                      # quota, leaderboard top 6, current best, queue
    python daily/run.py build NAME [extra blend args]   # base (best) args + overrides -> daily/out/NAME.csv
    python daily/run.py submit NAME "message"       # submit daily/out/NAME.csv, wait, log, update best
"""
import csv
import shutil
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
    """Override base flags with extra flags (flag followed by its values).

    Repeatable per-device flags (--fam-w FAMILY ..., --fam-pa-offset FAMILY ...) are keyed by flag + family, so an
    override replaces only that family's entry.
    """
    multi = {"--fam-w", "--fam-pa-offset"}

    def parse(tokens):
        groups, cur = [], None
        for tok in tokens:
            if tok.startswith("--"):
                cur = [tok]
                groups.append(cur)
            else:
                cur.append(tok)
        return {(g[0], g[1]) if g[0] in multi else (g[0],): g for g in groups}
    b, e = parse(base), parse(extra)
    b.update(e)
    return [x for g in b.values() for x in g]


def status():
    s = load()
    print(sh(["kaggle", "competitions", "submission-limits", COMP]).strip().splitlines()[-1])
    lb = [l for l in sh(["kaggle", "competitions", "leaderboard", COMP, "-s"]).splitlines() if l[:1].isdigit()][:8]
    print("leaderboard:", " | ".join(f"{i + 1}. {l.split()[-1]}" for i, l in enumerate(lb)))
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
    r = subprocess.run(["kaggle", "competitions", "submit", COMP, "-f", str(f), "-m", msg],
                       capture_output=True, text=True, cwd=UMUD)
    print(r.stdout.strip() or r.stderr.strip())
    for _ in range(90):  # up to 15 min
        time.sleep(10)
        top = next((l for l in sh(["kaggle", "competitions", "submissions", COMP]).splitlines() if f.name in l), "")
        if "COMPLETE" in top or "ERROR" in top:
            break
    if "COMPLETE" not in top:
        sys.exit(f"no score for {f.name}: {top.strip() or 'submission not found'}")
    score = float(top.split()[-1])
    shutil.copy(f, UMUD / "submissions" / f.name)  # permanent record of every submitted CSV
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
