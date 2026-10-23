#!/bin/bash
set -euo pipefail
cd /home/user/kraggle-3-in-one
export PYTHONPATH=$PWD OMP_NUM_THREADS=4
S='{"v12v11": [["t2h:v12_og_v3",0.175],["t2h:v12_og_v3_noloc",0.175],["v11_og_v3",0.175],["v11_og_v3_noloc",0.175],["t2h:v7_og_v2",0.15],["t2h:v7_og_v2_noloc",0.15]],
 "v11heavy": [["t2h:v12_og_v3",0.125],["t2h:v12_og_v3_noloc",0.125],["v11_og_v3",0.25],["v11_og_v3_noloc",0.25],["t2h:v7_og_v2",0.125],["t2h:v7_og_v2_noloc",0.125]]}'
T2_WORK=/home/user/work/t2 T2_FEAT=/home/user/work/t2/feat_v3 T2_PSEUDO_DIR=pseudo3 python3 -W ignore -m trafficflow.t2.pseudo score queue_ongoing "$S"
echo K4B_GATE_DONE
