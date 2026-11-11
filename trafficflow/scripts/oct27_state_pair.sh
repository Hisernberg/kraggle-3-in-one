#!/bin/bash
# 2026-10-27: H11P-recipe state (old) and the never-submitted H13P/H14P recipe (new: regular 0.25 P2 + 0.375 P3 + 0.375 P4,
# blackout rows mean(fullPD, fullPD2)) from the same rebuilt members, both through the production post-processing.
# The K-line state is then moved by (new - old), see oct27_state_delta.py.
source /home/urad/tfbroot/env.sh
export OMP_NUM_THREADS=4
W=/home/urad/tfbroot/work
SM="free=0.0075,free_a=0.001,gate=0.02,gate_a=0.005,dark=0.05"
python3 -W ignore -m trafficflow.t1_pipeline enskind --tag ens_old27 --reg fullP2:0.5 fullP3:0.5 --dark full7
python3 -W ignore -m trafficflow.t1_pipeline enskind --tag ens_new27 --reg fullP2:0.25 fullP3:0.375 fullP4:0.375 --dark fullPD:0.5 fullPD2:0.5
for t in old27 new27; do
  python3 -W ignore -m trafficflow.make_submission --state-tag ens_$t --recon-a 0.75 --gate 0.6 --smooth "$SM" \
    --queue $W/t2/lgb_v8og_v12v11.csv --odme $W/t4/work__t4__t4_l2proj.csv --out $W/subs/H_$t.csv --note "H_$t" &
done
wait
echo STATE_PAIR_DONE
