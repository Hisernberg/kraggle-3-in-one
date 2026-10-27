#!/bin/bash
# 2026-10-27: v14 = v12 recipe (og_v3 / og_v3_noloc, hybrid labels, t2h/feat_og) at p4 capacity, 4 threads, both in parallel.
source /home/urad/tfbroot/env.sh
L=/home/urad/tfbroot/logs
for v in og_v3 og_v3_noloc; do
T2_WORK=/home/urad/tfbroot/work/t2h T2_FEAT=/home/urad/tfbroot/work/t2h/feat_og python3 -W ignore -c "
from trafficflow.t2 import models
p, r = models.CFG['p4']; models.CFG['p4'] = ({**p, 'num_threads': 4}, r)
from trafficflow.t2.core import WORK
from trafficflow.t2.robust_pipeline import train_variant
f = WORK / 'model_v14_${v}_queue_ongoing.txt'
if not f.exists():
    train_variant('$v', 'p4', True, cond='queue_ongoing').save_model(str(f))
print('v14 $v ready', f, flush=True)
" > $L/v14_$v.log 2>&1 &
done
wait
echo V14_DONE
