#!/bin/bash
# 2026-10-27: March-adapt the remaining ongoing components of the v12v11 mix (v12 og_v3/noloc p3, v7 og_v2/noloc p2,
# hybrid-label t2h tables) so that April windows can be served by a fully adapted mix (K3 adapted only v11).
source /home/urad/tfbroot/env.sh
export OMP_NUM_THREADS=4
T="T2_WORK=/home/urad/tfbroot/work/t2h T2_FEAT=/home/urad/tfbroot/work/t2h/feat_og T2_PSEUDO_DIR=pseudo2"
L=/home/urad/tfbroot/logs
env $T python3 -W ignore -m trafficflow.t2.adapt train og_v3 p3 validation 1.0       > $L/ad_v12a.log 2>&1 &
env $T python3 -W ignore -m trafficflow.t2.adapt train og_v3_noloc p3 validation 1.0 > $L/ad_v12b.log 2>&1 &
env $T python3 -W ignore -m trafficflow.t2.adapt train og_v2 p2 validation 1.0       > $L/ad_v7a.log 2>&1 &
env $T python3 -W ignore -m trafficflow.t2.adapt train og_v2_noloc p2 validation 1.0 > $L/ad_v7b.log 2>&1 &
wait
echo ADAPT_ALL_DONE
