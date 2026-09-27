#!/bin/bash
# Add one seed-ensemble member to the Task 1 models and build the candidate submission if it passes the J gate.
#
#   bash trafficflow/seed_member.sh <seed> <hold_tag> <full_tag> <ens_tag> <prev_hold_tags,comma> <prev_full_tags,space> <out_id>
#   e.g. bash trafficflow/seed_member.sh 2 hold5 full5 ens345 hold3,hold4 "full3 full4" H3_t1ens345
#
# Gate (trafficflow/docs/LOOP.md): the ensemble with the new member improves holdout J (gate 0.6, a 0.75, TV smoothing)
# on at least 3 of 4 panels and on the mean, against the current ensemble. Log: /home/user/work/<out_id>.log
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD OMP_NUM_THREADS=2 TFB_FD=1 TFB_THREADS=2 TFB_PRED_THREADS=2
SEED=$1 HOLD=$2 FULL=$3 ENS=$4 PREV_HOLD=$5 PREV_FULL=$6 OUT=$7
W=/home/user/work
echo "== $(date -u +%H:%M) train $HOLD (seed $SEED)"
TFB_SEED=$SEED python3 -m trafficflow.t1_pipeline train --holdout --tag "$HOLD"
echo "== $(date -u +%H:%M) holdout predictions + ensemble evaluation"
python3 -m trafficflow.t1_smooth_eval preds "$HOLD"
python3 -m trafficflow.t1_smooth_eval ens "$PREV_HOLD,$HOLD" > "$W/t1/smooth/ens_$HOLD.txt"
python3 - "$PREV_HOLD" "$HOLD" <<'EOF'
import sys
import pandas as pd
prev, new = sys.argv[1].replace(",", "+"), sys.argv[1].replace(",", "+") + "+" + sys.argv[2]
d = pd.read_csv("/home/user/work/t1/smooth/ens.csv")
d = d[d.post == "gate+TV"].pivot_table(index="panel", columns="members", values="J")
dj = d[new] - d[prev]
print("dJ per panel:", dj.round(5).to_dict(), "mean", round(float(dj.mean()), 5))
ok = (dj > 0).sum() >= 3 and dj.mean() > 0
print("GATE", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 3)
EOF
echo "== $(date -u +%H:%M) train $FULL"
ROUNDS=$(python3 -m trafficflow.t1_pipeline rounds --tag "$HOLD")
TFB_SEED=$SEED python3 -m trafficflow.t1_pipeline train --tag "$FULL" --rounds "$ROUNDS"
echo "== $(date -u +%H:%M) predict $FULL"
python3 -m trafficflow.t1_pipeline predict --tag "$FULL"
python3 -m trafficflow.t1_pipeline ens --tag "$ENS" --members $PREV_FULL "$FULL"
echo "== $(date -u +%H:%M) build $OUT"
python3 -m trafficflow.make_submission --state-tag "$ENS" --recon-a 0.75 --gate 0.6 \
  --smooth "free=0.0075,free_a=0.001,gate=0.02,gate_a=0.005,dark=0.05" \
  --queue "$W/t2/lgb_v8_seeds9_stack03.csv" --odme "$W/t4/t4_l2proj.csv" \
  --out "$W/subs/$OUT.csv" --note "$OUT: Task 1 state = ensemble $ENS ($PREV_FULL $FULL)"
python3 -m trafficflow.loop pack "$W/subs/$OUT.csv"
echo "== $(date -u +%H:%M) done"
