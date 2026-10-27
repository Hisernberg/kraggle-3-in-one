#!/bin/bash
# 2026-10-27 L2 (private-line final candidate): L1 + April ongoing rows from the fully March-adapted v12v11 mix.
# Gate (oct27_gate_adapt.sh, pseudo3 official-style windows, April = honest test): fullAD +0.0237 S vs v12v11,
# +0.016 +- 0.004 mean window IoU vs aprAD (K3 recipe, v11 pair adapted only), 101 better / 43 worse.
# v14 (v12 recipe at p4) failed the gate (April -0.0023) and is not used.
set -euo pipefail
source /home/urad/tfbroot/env.sh
export OMP_NUM_THREADS=4
W=/home/urad/tfbroot/work; A=$W/t2h/adapt
T2_WORK=$W/t2 T2_FEAT=$W/t2/feat_v3 python3 -W ignore -m trafficflow.t2.ongoing_split_mix $W/t2/lgb_v8og_v12v11.csv $W/t2/lgb_v8og_v12v11_fullAD.csv \
 "{\"private\": [[\"$A:ad_og_v3_p3_validation_1.0\",0.175],[\"$A:ad_og_v3_noloc_p3_validation_1.0\",0.175],[\"adapt:ad_og_v3_p3_validation_1.0\",0.175],[\"adapt:ad_og_v3_noloc_p3_validation_1.0\",0.175],[\"$A:ad_og_v2_p2_validation_1.0\",0.15],[\"$A:ad_og_v2_noloc_p2_validation_1.0\",0.15]]}"
# April ongoing rows only; validation rows of L1 equal lgb_v8og_v12v11.csv (checked: splice --split validation is a no-op).
python3 -W ignore /home/urad/tfbroot/src/trafficflow/scripts/splice_ongoing.py $W/subs/L1_state.csv $W/t2/lgb_v8og_v12v11_fullAD.csv $W/subs/L2_fullAD.csv --split private
# -> 158 private queue cells differ from L1; zipped as L2_fullAD.zip with L1's checks (+ binary queue check).
