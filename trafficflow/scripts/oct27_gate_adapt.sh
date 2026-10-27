#!/bin/bash
# 2026-10-27 gate on pseudo3 (official-style windows): April = the honest test (adapted models train on March pseudo2).
# v12v11 (public line), aprAD (K3 recipe: v11 adapted), fullAD (every component adapted), and v14 (p4) if trained.
source /home/urad/tfbroot/env.sh
A=/home/urad/tfbroot/work/t2h/adapt
S='{"v12v11": [["t2h:v12_og_v3",0.175],["t2h:v12_og_v3_noloc",0.175],["v11_og_v3",0.175],["v11_og_v3_noloc",0.175],["t2h:v7_og_v2",0.15],["t2h:v7_og_v2_noloc",0.15]],
 "aprAD": [["t2h:v12_og_v3",0.175],["t2h:v12_og_v3_noloc",0.175],["adapt:ad_og_v3_p3_validation_1.0",0.175],["adapt:ad_og_v3_noloc_p3_validation_1.0",0.175],["t2h:v7_og_v2",0.15],["t2h:v7_og_v2_noloc",0.15]],
 "fullAD": [["'$A':ad_og_v3_p3_validation_1.0",0.175],["'$A':ad_og_v3_noloc_p3_validation_1.0",0.175],["adapt:ad_og_v3_p3_validation_1.0",0.175],["adapt:ad_og_v3_noloc_p3_validation_1.0",0.175],["'$A':ad_og_v2_p2_validation_1.0",0.15],["'$A':ad_og_v2_noloc_p2_validation_1.0",0.15]]'"${EXTRA:-}"'}'
T2_WORK=/home/urad/tfbroot/work/t2 T2_FEAT=/home/urad/tfbroot/work/t2/feat_v3 T2_PSEUDO_DIR=pseudo3 python3 -W ignore -m trafficflow.t2.pseudo score queue_ongoing "$S"
echo GATE_DONE
