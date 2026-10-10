#!/usr/bin/env bash
# Rebuild the WEAR working directory in a fresh container.
# Needs ~/.kaggle/access_token. Data and outputs are never committed to the repo.
set -euo pipefail
C=3rd-wear-dataset-challenge-hasca-2026
R=$(cd "$(dirname "$0")" && pwd)
D=/home/user/wear_data; W=/home/user/wear_work
pip install -q kaggle scikit-learn scipy >/dev/null 2>&1 || true
mkdir -p $D $W/own_subs $W/cands $W/kout
[ -f $D/test/test_meta_data.csv ] || { kaggle competitions download $C -p $D -q && (cd $D && unzip -q -o *.zip && rm -f *.zip); }
for k in wear-good-fork wear-good-fork2 wear-hanbat-gpu wear-fusion-full wear-uec-k1; do
  [ -d $W/kout/$k ] && [ "$(ls -A $W/kout/$k)" ] || { mkdir -p $W/kout/$k; kaggle kernels output koushikrudra/$k -p $W/kout/$k -q; }
done
cp $R/*.py $W/
cd $W && kaggle competitions submissions $C --page-size 200 -v > mysubs.csv && python3 dl_subs.py
python3 -c "import numpy as np; np.save('test_vmean.npy', np.load('$D/test/test_videomae_data.npy', mmap_mode='r').mean(-1))" 2>/dev/null || true
echo bootstrap done
