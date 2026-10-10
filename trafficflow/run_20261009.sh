#!/bin/bash
# 2026-10-09: restore (E0) and the five submissions H15b, H16b, H17, H18, H19, from a fresh container.
# Run from the repository root. Needs Kaggle credentials in the environment (never written to files).
# 4 cores / 15 GB: run the stages in order; at most two LightGBM trainings at once, and never two next to a
# pseudo-holdout scorer (an OOM kill hit exactly that combination on 2026-10-09).
# Wall time on 2026-10-09: ~3.5 h (downloads 10 min, caches 10 min, Task 2 chain 1 h, v12 + adapt 1 h, scoring 1 h).
set -euo pipefail
export PYTHONPATH=$PWD OMP_NUM_THREADS=2
SMOOTH="free=0.0075,free_a=0.001,gate=0.02,gate_a=0.005,dark=0.05"
mk() {  # mk <ID> <state tag> <queue csv> <note>
  python3 -m trafficflow.make_submission --state-tag "$2" --recon-a 0.75 --gate 0.6 --smooth "$SMOOTH" \
    --queue "$3" --odme /home/user/work/t4/t4_l2proj.csv --out "/home/user/work/subs/$1.csv" --note "$4"
  python3 -m trafficflow.loop pack "/home/user/work/subs/$1.csv"
}

# ---- E0: data, caches, backup ---------------------------------------------------------------------------------
# kaggle competitions download -c 2026-ieee-big-data-traffic-flow-bench -p /home/user/data   (unzip -> kaggle_public, delete zip)
# kaggle datasets download kragglenote2forwork/tfb-work -p /home/user/work/backup --unzip
#   then copy each file to its path in manifest.json (the H10P/H11P zips unpack to folders holding the CSV)
python3 -m trafficflow.data                                   # dense caches, ~10 min on 2 processes
# H11P from the backed-up state / queue / ODME: byte-identical to the backed-up CSV (md5 2ed0d98d...)
mk H11P_rebuild ens_H11P /home/user/work/t2/lgb_v8_seeds9_stack03.csv "H11P rebuild (verification)"

# ---- H15b: month speed bias (Task 1 regular rows) ---------------------------------------------------------------
python3 -m trafficflow.t1_pseudo build                        # pseudo cells, 40k per panel, ~4 min
python3 -c "from trafficflow.t1_pseudo import predict; predict(['fullP2', 'fullP3'], threads=2)"
python3 -m trafficflow.t1_bias check fullP2:0.5,fullP3:0.5    # +0.00010 Mar / +0.00012 Apr, 10/10 panels
python3 -m trafficflow.t1_bias apply fullP2:0.5,fullP3:0.5 ens_H11P ens_H15b
mk H15b_monthbias ens_H15b /home/user/work/t2/lgb_v8_seeds9_stack03.csv "H15b: H11P + month speed bias"

# ---- Task 2 components (v7, v11, v12) and feature tables --------------------------------------------------------
python3 -W ignore -m trafficflow.t2.dataset
T2_FEAT=/home/user/work/t2/feat_v3 T2_PHYSICS=1 python3 -W ignore -m trafficflow.t2.build_features
T2_FEAT=/home/user/work/t2/feat_v3 python3 -W ignore -m trafficflow.t2.og_components v11
python3 -W ignore -m trafficflow.t2.truthfix fit
python3 -W ignore -m trafficflow.t2.truthfix relabel hybrid
T2_WORK=/home/user/work/t2h T2_TRUTHQ=/home/user/work/t2/ds_{panel}_y2.npz python3 -W ignore -m trafficflow.t2.dataset
T2_WORK=/home/user/work/t2h T2_PHYSICS=1 T2_ONLY=queue_ongoing T2_FEAT=/home/user/work/t2h/feat_og \
  python3 -W ignore -m trafficflow.t2.build_features
T2_WORK=/home/user/work/t2h T2_FEAT=/home/user/work/t2h/feat_og python3 -W ignore -m trafficflow.t2.og_components v7
T2_WORK=/home/user/work/t2h T2_FEAT=/home/user/work/t2h/feat_og python3 -W ignore -m trafficflow.t2.og_components v12

# ---- H16b / H17: public-line ongoing mixes ----------------------------------------------------------------------
export T2_WORK=/home/user/work/t2 T2_FEAT=/home/user/work/t2/feat_v3
python3 -W ignore -m trafficflow.t2.ongoing_mix v7v11 /home/user/work/t2/lgb_v8og_v7v11.csv     # 105 + 88 cells vs v5
mk H16b_monthbias_og_v7v11 ens_H15b /home/user/work/t2/lgb_v8og_v7v11.csv "H16b: H15b + ongoing v7+v11"
python3 -W ignore -m trafficflow.t2.ongoing_mix v12v11 /home/user/work/t2/lgb_v8og_v12v11.csv   # 37 + 31 cells vs v7v11
mk H17_og_v12v11 ens_H15b /home/user/work/t2/lgb_v8og_v12v11.csv "H17: H16b + ongoing v12 replacing v7 og_v3/noloc"

# ---- Task 2 pseudo-holdout tables and April adaptation -----------------------------------------------------------
T2_PHYSICS=1 T2_PSEUDO_MODE=sim T2_PSEUDO_DIR=pseudo3 python3 -W ignore -m trafficflow.t2.pseudo build   # official-style
T2_PHYSICS=1 T2_PSEUDO_MODE=all T2_PSEUDO_DIR=pseudo2 python3 -W ignore -m trafficflow.t2.pseudo build   # all windows
T2_PSEUDO_DIR=pseudo2 python3 -W ignore -m trafficflow.t2.adapt train og_v3 p3 validation 1.0           # ~35 min each
T2_PSEUDO_DIR=pseudo2 python3 -W ignore -m trafficflow.t2.adapt train og_v3_noloc p3 validation 1.0
# scoring (6 min on pseudo3, ~30 min and ~6 GB on pseudo2; run alone):
#   T2_PSEUDO_DIR=pseudo3 python3 -m trafficflow.t2.pseudo score queue_ongoing "$(cat schemes.json)"
#   schemes as in docs/EXPERIMENTS.md, 2026-10-09 section

# ---- H18 / H19: private line (April rows only; March rows identical to the parent) ------------------------------
AD='["adapt:ad_og_v3_p3_validation_1.0",0.175],["adapt:ad_og_v3_noloc_p3_validation_1.0",0.175],["t2h:v7_og_v2",0.15],["t2h:v7_og_v2_noloc",0.15]'
python3 -W ignore -m trafficflow.t2.ongoing_split_mix /home/user/work/t2/lgb_v8og_v7v11.csv \
  /home/user/work/t2/lgb_v8og_v7v11_aprAD.csv "{\"private\": [[\"t2h:v7_og_v3\",0.175],[\"t2h:v7_og_v3_noloc\",0.175],$AD]}"
mk H18_aprAdapt ens_H15b /home/user/work/t2/lgb_v8og_v7v11_aprAD.csv "H18: H16b + April ongoing adapted"
python3 -W ignore -m trafficflow.t2.ongoing_split_mix /home/user/work/t2/lgb_v8og_v12v11.csv \
  /home/user/work/t2/lgb_v8og_v12v11_aprAD.csv "{\"private\": [[\"t2h:v12_og_v3\",0.175],[\"t2h:v12_og_v3_noloc\",0.175],$AD]}"
mk H19_H17_aprAdapt ens_H15b /home/user/work/t2/lgb_v8og_v12v11_aprAD.csv "H19: H17 + April ongoing adapted"

# every file: python3 -m trafficflow.loop diff <parent>.zip <file>.zip  (only the intended rows change)
