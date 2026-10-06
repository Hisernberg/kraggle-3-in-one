#!/bin/bash
# Regular-only Task 1 member: the regular-cell models only (TFB_KINDS=reg). Blackout rows stay with the current
# members. Gated on holdout J with the blackout rows fixed, so the gate measures the regular-cell change alone, which
# is the part that transfers to the LB exactly (trafficflow/docs/EXPERIMENTS.md, 2026-09-27).
#
#   bash trafficflow/reg_member.sh <seed> <hold_tag> <full_tag> <reg_hold,comma> <dark_hold,comma> \
#        "<reg_full space>" "<dark_full space>" <out_id>
#   e.g. TFB_NREG=300000 TFB_LR=0.05 TFB_MAXR=6000 bash trafficflow/reg_member.sh 6 hold9 full9 \
#        hold3,hold4,hold5,hold7 hold7 "full3 full4 full5 full7" "full7" H7_reg9
#
# Gate: the best of {add (equal member), half (new member weight 0.5), alone} improves holdout J over the current
# regular set on at least 3 of 4 panels and on the mean. Log: /home/user/work/<out_id>.log
set -euo pipefail
cd "$(dirname "$0")/.."
TH=${TFB_THREADS:-4}
export PYTHONPATH=$PWD OMP_NUM_THREADS=$TH TFB_FD=1 TFB_KINDS=reg TFB_THREADS=$TH TFB_PRED_THREADS=$TH
SEED=$1 HOLD=$2 FULL=$3 REGH=$4 DARKH=$5 REGF=$6 DARKF=$7 OUT=$8
W=/home/user/work
echo "== $(date -u +%H:%M) train $HOLD (regular models, seed $SEED, nreg ${TFB_NREG:-150000}, lr ${TFB_LR:-0.1}, maxr ${TFB_MAXR:-3000})"
TFB_SEED=$SEED python3 -m trafficflow.t1_pipeline train --holdout --tag "$HOLD"
echo "== $(date -u +%H:%M) holdout predictions + evaluation"
python3 -m trafficflow.t1_smooth_eval preds "$HOLD"
python3 -m trafficflow.t1_weighted_eval regs "$HOLD" "$REGH" "$DARKH"
PICK=$(python3 - <<'PY'
import sys
import pandas as pd
d = pd.read_csv("/home/user/work/t1/smooth/regs.csv").pivot_table(index="panel", columns="reg", values="J")
best, bj = None, 0.0
for s in ("add", "half", "alone"):
    dj = d[s] - d["base"]
    print(f"{s}: dJ per panel {dj.round(5).to_dict()} mean {dj.mean():.5f}", file=sys.stderr)
    if (dj > 0).sum() >= 3 and dj.mean() > bj:
        best, bj = s, dj.mean()
print("GATE", "PASS " + best if best else "FAIL", file=sys.stderr)
print(best or "")
PY
)
[ -n "$PICK" ] || exit 3
case "$PICK" in
  add) REGSET="$REGF $FULL" ;;
  half) REGSET="$REGF $(for _ in $REGF; do printf '%s ' "$FULL"; done)" ;;   # equal mean of old + n copies = 0.5 / 0.5
  alone) REGSET="$FULL" ;;
esac
echo "== $(date -u +%H:%M) picked $PICK -> regular rows = mean($REGSET); train $FULL"
ROUNDS=$(python3 -m trafficflow.t1_pipeline rounds --tag "$HOLD")
TFB_SEED=$SEED python3 -m trafficflow.t1_pipeline train --tag "$FULL" --rounds "$ROUNDS"
echo "== $(date -u +%H:%M) predict $FULL (regular rows)"
python3 -m trafficflow.t1_pipeline predict --tag "$FULL"
python3 -m trafficflow.t1_pipeline enskind --tag "ens_$OUT" --reg $REGSET --dark $DARKF
echo "== $(date -u +%H:%M) build $OUT"
python3 -m trafficflow.make_submission --state-tag "ens_$OUT" --recon-a 0.75 --gate 0.6 \
  --smooth "free=0.0075,free_a=0.001,gate=0.02,gate_a=0.005,dark=0.05" \
  --queue "$W/t2/lgb_v8_seeds9_stack03.csv" --odme "$W/t4/t4_l2proj.csv" \
  --out "$W/subs/$OUT.csv" --note "$OUT: Task 1 regular rows = mean($REGSET), blackout rows = mean($DARKF)"
python3 -m trafficflow.loop pack "$W/subs/$OUT.csv"
echo "== $(date -u +%H:%M) done"
