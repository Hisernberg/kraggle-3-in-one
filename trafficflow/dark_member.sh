#!/bin/bash
# Dark-only Task 1 member: blackout ("dark") models only (TFB_KINDS=dark), with ramp-flow features. Regular rows
# of the submission come from the regular members; blackout rows from the best dark set that passes the J gate.
#
#   bash trafficflow/dark_member.sh <seed> <hold_tag> <full_tag> <reg_hold,comma> <ref_dark_hold> \
#        "<reg_full space>" <ref_dark_full> <out_id>
#   e.g. bash trafficflow/dark_member.sh 5 hold8 full8 hold3,hold4,hold5,hold7 hold7 "full3 full4 full5 full7" full7 H6_dark8
#
# Capacity knobs are passed through the environment (TFB_DARK_LEAVES, TFB_DARK_LR, TFB_MAXR, TFB_NDARK).
# Gate (trafficflow/docs/LOOP.md): the best of {new, ref+new} on blackout rows improves holdout J (gate 0.6, a 0.75,
# TV smoothing) over ref alone on at least 3 of 4 panels and on the mean. Log: /home/user/work/<out_id>.log
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD OMP_NUM_THREADS=2 TFB_FD=1 TFB_RAMP=1 TFB_KINDS=dark TFB_THREADS=2 TFB_PRED_THREADS=2
SEED=$1 HOLD=$2 FULL=$3 REGH=$4 REFH=$5 REGF=$6 REFF=$7 OUT=$8
W=/home/user/work
echo "== $(date -u +%H:%M) train $HOLD (dark models, seed $SEED, leaves ${TFB_DARK_LEAVES:-63}, lr ${TFB_DARK_LR:-0.05}, maxr ${TFB_MAXR:-3000}, ndark ${TFB_NDARK:-150000})"
TFB_SEED=$SEED python3 -m trafficflow.t1_pipeline train --holdout --tag "$HOLD"
echo "== $(date -u +%H:%M) holdout predictions + per-kind evaluation"
python3 -m trafficflow.t1_smooth_eval preds "$HOLD"
python3 -m trafficflow.t1_weighted_eval kinds "$REGH" "$REFH;$HOLD;$REFH+$HOLD"
PICK=$(python3 - "$REFH" "$HOLD" <<'PY'
import sys
import pandas as pd
ref, new = sys.argv[1], sys.argv[2]
d = pd.read_csv("/home/user/work/t1/smooth/kinds.csv").pivot_table(index="panel", columns="dark", values="J")
best, bj = None, 0.0
for s in (new, f"{ref}+{new}"):
    dj = d[s] - d[ref]
    print(f"{s}: dJ per panel {dj.round(5).to_dict()} mean {dj.mean():.5f}", file=sys.stderr)
    if (dj > 0).sum() >= 3 and dj.mean() > bj:
        best, bj = s, dj.mean()
print("GATE", "PASS " + best if best else "FAIL", file=sys.stderr)
print(best or "")
PY
)
[ -n "$PICK" ] || exit 3
DARKF=$(echo "$PICK" | sed "s/$REFH/$REFF/; s/$HOLD/$FULL/; s/+/ /")
echo "== $(date -u +%H:%M) picked dark set $PICK -> $DARKF; train $FULL"
ROUNDS=$(python3 -m trafficflow.t1_pipeline rounds --tag "$HOLD")
TFB_SEED=$SEED python3 -m trafficflow.t1_pipeline train --tag "$FULL" --rounds "$ROUNDS"
echo "== $(date -u +%H:%M) predict $FULL (dark rows)"
python3 -m trafficflow.t1_pipeline predict --tag "$FULL"
python3 -m trafficflow.t1_pipeline enskind --tag "ens_$OUT" --reg $REGF --dark $DARKF
echo "== $(date -u +%H:%M) build $OUT"
python3 -m trafficflow.make_submission --state-tag "ens_$OUT" --recon-a 0.75 --gate 0.6 \
  --smooth "free=0.0075,free_a=0.001,gate=0.02,gate_a=0.005,dark=0.05" \
  --queue "$W/t2/lgb_v8_seeds9_stack03.csv" --odme "$W/t4/t4_l2proj.csv" \
  --out "$W/subs/$OUT.csv" --note "$OUT: Task 1 regular rows = mean($REGF), blackout rows = mean($DARKF)"
python3 -m trafficflow.loop pack "$W/subs/$OUT.csv"
echo "== $(date -u +%H:%M) done"
