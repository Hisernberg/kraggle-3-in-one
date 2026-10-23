#!/bin/bash
set -euo pipefail
cd /home/user/kraggle-3-in-one
export PYTHONPATH=$PWD OMP_NUM_THREADS=4
mk=/home/user/work/.chain
[ -f $mk/k4_seed1 ] || { T2_WORK=/home/user/work/t2h T2_FEAT=/home/user/work/t2h/feat_og python3 -W ignore -m trafficflow.t2.og_components v12 1; touch $mk/k4_seed1; }
[ -f $mk/k4_p3 ] || { T2_WORK=/home/user/work/t2 T2_FEAT=/home/user/work/t2/feat_v3 T2_PHYSICS=1 T2_PSEUDO_MODE=sim T2_PSEUDO_DIR=pseudo3 python3 -W ignore -m trafficflow.t2.pseudo build; touch $mk/k4_p3; }
S='{"v12v11": [["t2h:v12_og_v3",0.175],["t2h:v12_og_v3_noloc",0.175],["v11_og_v3",0.175],["v11_og_v3_noloc",0.175],["t2h:v7_og_v2",0.15],["t2h:v7_og_v2_noloc",0.15]],
 "v12x2v11": [["t2h:v12_og_v3",0.0875],["t2h:v12_og_v3_s1",0.0875],["t2h:v12_og_v3_noloc",0.0875],["t2h:v12_og_v3_noloc_s1",0.0875],["v11_og_v3",0.175],["v11_og_v3_noloc",0.175],["t2h:v7_og_v2",0.15],["t2h:v7_og_v2_noloc",0.15]]}'
T2_WORK=/home/user/work/t2 T2_FEAT=/home/user/work/t2/feat_v3 T2_PSEUDO_DIR=pseudo3 python3 -W ignore -m trafficflow.t2.pseudo score queue_ongoing "$S"
echo K4_GATE_DONE
