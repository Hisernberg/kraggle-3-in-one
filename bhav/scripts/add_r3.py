"""Add round-3 head outputs (r3_<backbone>.npz) and the fusion head to the member pool -> work/probs/members_r3.pkl.

New keys per backbone b (short name s): ML_s (multi-layer head), B_s (bagged last-layer head: round-2 seed + round-3
seed, geometric mean), BT_s (B_s with the round-3 seed's TTA prediction added), FUS (fusion head, seeds a+b).
"""
import sys, os, glob, pickle
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from common import *
import blend

SHORT = {'whisper_large_v2': 'large_v2', 'whisper_large_v1': 'large_v1', 'whisper_vaani_large_v3': 'vaani_large_v3',
         'whisper_hi_medium': 'hi_medium'}
gm = lambda *ps: np.exp(np.mean([np.log(p + 1e-9) for p in ps], 0))

if __name__ == '__main__':
    m = load_meta(); R = pickle.load(open(f'{BH}/work/probs/members_heads.pkl', 'rb'))
    for f in sorted(glob.glob(f'{BH}/work/r3/r3_*.npz')):
        b = os.path.basename(f)[3:-4]; s = SHORT[b]; z = np.load(f); old = R[f'A_{s}']
        R[f'ML_{s}'] = (z['oof_ml'], z['test_ml'])
        R[f'B_{s}'] = (gm(old[0], z['oof_sl2']), gm(old[1], z['test_sl2']))
        R[f'BT_{s}'] = (gm(old[0], z['oof_sl2'], z['oof_sl2_tta']), gm(old[1], z['test_sl2'], z['test_sl2_tta']))
    f = f'{BH}/work/r3/fusion.npz'
    if os.path.exists(f):
        z = np.load(f); R['FUS'] = (gm(z['oof_a'], z['oof_b']), gm(z['test_a'], z['test_b']))
    pickle.dump(R, open(f'{BH}/work/probs/members_r3.pkl', 'wb'))
    for k in R:
        if k.split('_')[0] in ('A', 'ML', 'B', 'BT', 'FUS'):
            print(f'{k:20s}', blend.evaluate(m, R[k][0]).round(4), flush=True)
