#!/bin/bash
set -euo pipefail
cd /home/user/kraggle-3-in-one
export PYTHONPATH=$PWD OMP_NUM_THREADS=2
python3 -W ignore -m trafficflow.t2.dataset
T2_FEAT=/home/user/work/t2/feat_v3 T2_PHYSICS=1 python3 -W ignore -m trafficflow.t2.build_features
T2_FEAT=/home/user/work/t2/feat_v3 python3 -W ignore -m trafficflow.t2.og_components v11
python3 -W ignore -m trafficflow.t2.truthfix fit
python3 -W ignore -m trafficflow.t2.truthfix relabel hybrid
T2_WORK=/home/user/work/t2h T2_TRUTHQ=/home/user/work/t2/ds_{panel}_y2.npz python3 -W ignore -m trafficflow.t2.dataset
T2_WORK=/home/user/work/t2h T2_PHYSICS=1 T2_ONLY=queue_ongoing T2_FEAT=/home/user/work/t2h/feat_og python3 -W ignore -m trafficflow.t2.build_features
T2_WORK=/home/user/work/t2h T2_FEAT=/home/user/work/t2h/feat_og python3 -W ignore -m trafficflow.t2.og_components v7
T2_WORK=/home/user/work/t2h T2_FEAT=/home/user/work/t2h/feat_og python3 -W ignore -m trafficflow.t2.og_components v12
export T2_WORK=/home/user/work/t2 T2_FEAT=/home/user/work/t2/feat_v3
python3 -W ignore -m trafficflow.t2.ongoing_mix v12v11 /home/user/work/t2/lgb_v8og_v12v11.csv
echo CHAIN_V12_DONE
T2_PHYSICS=1 T2_PSEUDO_MODE=all T2_PSEUDO_DIR=pseudo2 python3 -W ignore -m trafficflow.t2.pseudo build
T2_PSEUDO_DIR=pseudo2 python3 -W ignore -m trafficflow.t2.adapt train og_v3 p3 validation 1.0
T2_PSEUDO_DIR=pseudo2 python3 -W ignore -m trafficflow.t2.adapt train og_v3_noloc p3 validation 1.0
AD='["adapt:ad_og_v3_p3_validation_1.0",0.175],["adapt:ad_og_v3_noloc_p3_validation_1.0",0.175],["t2h:v7_og_v2",0.15],["t2h:v7_og_v2_noloc",0.15]'
python3 -W ignore -m trafficflow.t2.ongoing_split_mix /home/user/work/t2/lgb_v8og_v12v11.csv \
  /home/user/work/t2/lgb_v8og_v12v11_aprAD.csv "{\"private\": [[\"t2h:v12_og_v3\",0.175],[\"t2h:v12_og_v3_noloc\",0.175],$AD]}"
echo CHAIN_ADAPT_DONE
