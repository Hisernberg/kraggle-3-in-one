#!/bin/bash
# Resumable: each finished step leaves a marker in /home/user/work/.chain/
set -euo pipefail
mkdir -p /home/user/work/t2/feat_v3 /home/user/work/t2h/feat_og /home/user/work/t2h/feat /home/user/work/.chain
cd /home/user/kraggle-3-in-one
export PYTHONPATH=$PWD OMP_NUM_THREADS=${THREADS:-4}
step() { local n=$1; shift; if [ -f /home/user/work/.chain/$n ]; then echo "skip $n"; return; fi; echo "== $n $(date -u +%H:%M)"; env "$@"; touch /home/user/work/.chain/$n; }
step 01_ds        python3 -W ignore -m trafficflow.t2.dataset
step 02_feat_v3   T2_FEAT=/home/user/work/t2/feat_v3 T2_PHYSICS=1 python3 -W ignore -m trafficflow.t2.build_features
step 03_v11       T2_FEAT=/home/user/work/t2/feat_v3 python3 -W ignore -m trafficflow.t2.og_components v11
step 04_tf_fit    python3 -W ignore -m trafficflow.t2.truthfix fit
step 05_tf_relab  python3 -W ignore -m trafficflow.t2.truthfix relabel hybrid
step 06_t2h_ds    T2_WORK=/home/user/work/t2h T2_TRUTHQ=/home/user/work/t2/ds_{panel}_y2.npz python3 -W ignore -m trafficflow.t2.dataset
step 07_t2h_feat  T2_WORK=/home/user/work/t2h T2_PHYSICS=1 T2_ONLY=queue_ongoing T2_FEAT=/home/user/work/t2h/feat_og python3 -W ignore -m trafficflow.t2.build_features
step 08_v7        T2_WORK=/home/user/work/t2h T2_FEAT=/home/user/work/t2h/feat_og python3 -W ignore -m trafficflow.t2.og_components v7
step 09_v12       T2_WORK=/home/user/work/t2h T2_FEAT=/home/user/work/t2h/feat_og python3 -W ignore -m trafficflow.t2.og_components v12
step 10_mix       T2_WORK=/home/user/work/t2 T2_FEAT=/home/user/work/t2/feat_v3 python3 -W ignore -m trafficflow.t2.ongoing_mix v12v11 /home/user/work/t2/lgb_v8og_v12v11.csv
echo CHAIN_V12_DONE
step 11_pseudo2   T2_WORK=/home/user/work/t2 T2_FEAT=/home/user/work/t2/feat_v3 T2_PHYSICS=1 T2_PSEUDO_MODE=all T2_PSEUDO_DIR=pseudo2 python3 -W ignore -m trafficflow.t2.pseudo build
step 12_ad1       T2_WORK=/home/user/work/t2 T2_FEAT=/home/user/work/t2/feat_v3 T2_PSEUDO_DIR=pseudo2 python3 -W ignore -m trafficflow.t2.adapt train og_v3 p3 validation 1.0
step 13_ad2       T2_WORK=/home/user/work/t2 T2_FEAT=/home/user/work/t2/feat_v3 T2_PSEUDO_DIR=pseudo2 python3 -W ignore -m trafficflow.t2.adapt train og_v3_noloc p3 validation 1.0
AD='["adapt:ad_og_v3_p3_validation_1.0",0.175],["adapt:ad_og_v3_noloc_p3_validation_1.0",0.175],["t2h:v7_og_v2",0.15],["t2h:v7_og_v2_noloc",0.15]'
step 14_split     T2_WORK=/home/user/work/t2 T2_FEAT=/home/user/work/t2/feat_v3 python3 -W ignore -m trafficflow.t2.ongoing_split_mix /home/user/work/t2/lgb_v8og_v12v11.csv \
  /home/user/work/t2/lgb_v8og_v12v11_aprAD.csv "{\"private\": [[\"t2h:v12_og_v3\",0.175],[\"t2h:v12_og_v3_noloc\",0.175],$AD]}"
echo CHAIN_ADAPT_DONE
