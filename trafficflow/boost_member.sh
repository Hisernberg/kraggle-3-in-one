#!/bin/bash
# Another regular-only boosted Task 1 member (TFB_KINDS=reg), gated against a JSON map of regular-row weightings.
#
#   bash trafficflow/boost_member.sh <seed> <hold_tag> <full_tag> <dark_hold,comma> "<dark_full space>" <out_id> '<spec>'
#   spec = {"base": [["hold9", 0.5], ["hold3", 0.125], ...], "cand1": [...], ...}   (the first entry is the reference;
#          hold tags only; the full-data members are the same tags with "hold" -> "full")
#   Capacity knobs through the environment: TFB_NREG, TFB_LR, TFB_MAXR, TFB_THREADS (default 4).
#
# Gate: the best non-reference weighting improves holdout J (blackout rows fixed) on at least 3 of 4 panels and on
# the mean. Log: /home/user/work/<out_id>.log
set -euo pipefail
cd "$(dirname "$0")/.."
TH=${TFB_THREADS:-4}
export PYTHONPATH=$PWD OMP_NUM_THREADS=$TH TFB_FD=1 TFB_KINDS=reg TFB_THREADS=$TH TFB_PRED_THREADS=$TH
SEED=$1 HOLD=$2 FULL=$3 DARKH=$4 DARKF=$5 OUT=$6 SPEC=$7
W=/home/user/work
echo "== $(date -u +%H:%M) train $HOLD (regular models, seed $SEED, nreg ${TFB_NREG:-150000}, lr ${TFB_LR:-0.1}, maxr ${TFB_MAXR:-3000})"
TFB_SEED=$SEED python3 -m trafficflow.t1_pipeline train --holdout --tag "$HOLD"
echo "== $(date -u +%H:%M) holdout predictions + evaluation"
python3 -m trafficflow.t1_smooth_eval preds "$HOLD"
python3 -m trafficflow.t1_weighted_eval regspec "$SPEC" "$DARKH"
REGW=$(python3 - "$SPEC" <<'PY'
import json, sys
import pandas as pd
spec = json.loads(sys.argv[1]); ref = list(spec)[0]
d = pd.read_csv("/home/user/work/t1/smooth/regs.csv").pivot_table(index="panel", columns="reg", values="J")
best, bj = None, 0.0
for s in list(spec)[1:]:
    dj = d[s] - d[ref]
    print(f"{s}: dJ per panel {dj.round(5).to_dict()} mean {dj.mean():.5f}", file=sys.stderr)
    if (dj > 0).sum() >= 3 and dj.mean() > bj:
        best, bj = s, dj.mean()
print("GATE", f"PASS {best}" if best else "FAIL", file=sys.stderr)
print(" ".join(f"{t.replace('hold', 'full')}:{w}" for t, w in spec[best]) if best else "")
PY
)
[ -n "$REGW" ] || exit 3
echo "== $(date -u +%H:%M) regular rows = $REGW; train $FULL"
ROUNDS=$(python3 -m trafficflow.t1_pipeline rounds --tag "$HOLD")
TFB_SEED=$SEED python3 -m trafficflow.t1_pipeline train --tag "$FULL" --rounds "$ROUNDS"
echo "== $(date -u +%H:%M) predict $FULL (regular rows)"
python3 -m trafficflow.t1_pipeline predict --tag "$FULL"
python3 -m trafficflow.t1_pipeline enskind --tag "ens_$OUT" --reg $REGW --dark $DARKF
echo "== $(date -u +%H:%M) build $OUT"
python3 -m trafficflow.make_submission --state-tag "ens_$OUT" --recon-a 0.75 --gate 0.6 \
  --smooth "free=0.0075,free_a=0.001,gate=0.02,gate_a=0.005,dark=0.05" \
  --queue "$W/t2/lgb_v8_seeds9_stack03.csv" --odme "$W/t4/t4_l2proj.csv" \
  --out "$W/subs/$OUT.csv" --note "$OUT: Task 1 regular rows = $REGW, blackout rows = mean($DARKF)"
python3 -m trafficflow.loop pack "$W/subs/$OUT.csv"
echo "== $(date -u +%H:%M) done"
