#!/bin/bash
# 2026-10-28 Task 1 line (gate: test-month pseudo-holdout, which has transferred at 1.2-2x on every transductive step).
# Q1 = L1 + month speed level (t1_bias, cross-fitted per panel x link x month x speed band) via the K-line delta.
source /home/urad/tfbroot/env.sh
W=/home/urad/tfbroot/work; SM="free=0.0075,free_a=0.001,gate=0.02,gate_a=0.005,dark=0.05"
python3 -c "from trafficflow.t1_pseudo import predict; predict(['fullP2','fullP3','fullP4'], threads=4)"
python3 -m trafficflow.t1_bias check fullP2:0.25,fullP3:0.375,fullP4:0.375        # +0.00010 Mar / +0.00012 Apr, 10/10
python3 -m trafficflow.t1_bias apply fullP2:0.25,fullP3:0.375,fullP4:0.375 ens_new27 ens_new27b
python3 -m trafficflow.make_submission --state-tag ens_new27b --recon-a 0.75 --gate 0.6 --smooth "$SM" \
  --queue $W/t2/lgb_v8og_v12v11.csv --odme $W/t4/work__t4__t4_l2proj.csv --out $W/subs/H_new27b.csv --note H_new27b
python3 /home/urad/work/k3/eb/trafficflow/scripts/oct27_state_delta.py $W/subs/L1_state.csv $W/subs/H_new27.csv \
  $W/subs/H_new27b.csv $W/subs/Q1_monthbias.csv
# more members (variance reduction): fullP5 / fullP6 (4x transductive rows, seeds 17/18), fullPD3 (dark+dark2, seed 19)
bash /home/urad/tfbroot/t1train.sh fullP5 17 reg 6 TFB_PSEUDO=1 TFB_PSEUDO_FILES=trainrows,trainrows2,trainrows3,trainrows4
bash /home/urad/tfbroot/t1train.sh fullP6 18 reg 6 TFB_PSEUDO=1 TFB_PSEUDO_FILES=trainrows4,trainrows3,trainrows2,trainrows
bash /home/urad/tfbroot/t1train.sh fullPD3 19 dark 4 TFB_RAMP=1 TFB_PSEUDO_DARK=1 TFB_PSEUDO_DARK_FILES=dark,dark2
# Q2: equal mix of P2..P6 + month bias refit, moved onto Q1 by (H_new28b - H_new27b)
python3 -c "from trafficflow.t1_pseudo import predict; predict(['fullP5','fullP6'], threads=4)"
python3 -m trafficflow.t1_pipeline enskind --tag ens_new28 --reg fullP2:0.2 fullP3:0.2 fullP4:0.2 fullP5:0.2 fullP6:0.2 --dark fullPD:0.5 fullPD2:0.5
python3 -m trafficflow.t1_bias apply fullP2:0.2,fullP3:0.2,fullP4:0.2,fullP5:0.2,fullP6:0.2 ens_new28 ens_new28b
python3 -m trafficflow.make_submission --state-tag ens_new28b --recon-a 0.75 --gate 0.6 --smooth "$SM" \
  --queue $W/t2/lgb_v8og_v12v11.csv --odme $W/t4/work__t4__t4_l2proj.csv --out $W/subs/H_new28b.csv --note H_new28b
python3 /home/urad/work/k3/eb/trafficflow/scripts/oct27_state_delta.py $W/subs/Q1_monthbias.csv $W/subs/H_new27b.csv \
  $W/subs/H_new28b.csv $W/subs/Q2_P2to6.csv
# Q3 = Q2 + April ongoing from fullAD (private line)
python3 /home/urad/tfbroot/src/trafficflow/scripts/splice_ongoing.py $W/subs/Q2_P2to6.csv $W/t2/lgb_v8og_v12v11_fullAD.csv \
  $W/subs/Q3_P2to6_fullAD.csv --split private
