"""Compute 3-seed OOF + test probabilities for the candidate blend members -> work/probs/members.pkl,
then greedy forward selection (with replacement) of an equal-weight log-prob blend on CV novel macro-F1."""
import sys, os, pickle
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from common import *
import blend

MEMBERS = {
    'wl3': dict(model='whisper_large_v3', layers=[32], std=True, C=0.05),
    'wl1': dict(model='whisper_large_v1', layers=[32], std=False, C=0.05),
    'wl2': dict(model='whisper_large_v2', layers=[32], std=True, C=0.05),
    'vaani': dict(model='whisper_vaani_large_v3', layers=[32], std=False, C=0.05),
    'hil2': dict(model='whisper_hi_large_v2', layers=[32], std=True, C=0.05),
    'colab': dict(model='whisper_collabora_hi_large_v2', layers=[32], std=False, C=0.05),
    'hingl': dict(model='whisper_hinglish_prime', layers=[31], std=True, C=0.05),
    'him': dict(model='whisper_hi_medium', layers=[23], std=True, C=0.05),
    'wm': dict(model='whisper_medium', layers=[24], std=False, C=0.1),
    'xlsr': dict(model='xlsr_300m', layers=[18], std=True, C=0.05),
    'w2vb': dict(model='w2vbert2', layers=[11], std=True, C=0.05),
    'mms': dict(model='mms_300m', layers=[15], std=True, C=0.05),
    'wavlm': dict(model='wavlm_large', layers=[8], std=True, C=0.05),
    'w2vb16': dict(model='w2vbert2', layers=[16], std=True, C=0.05),
    'w2vb_mid': dict(model='w2vbert2', layers=[11, 13, 16], std=True, C=0.05),
    'aud7': dict(model='audeering_dim', layers=[7], std=True, C=0.05),
    'hub16': dict(model='hubert_large', layers=[16], std=True, C=0.05),
    'wl1_s': dict(model='whisper_large_v1', layers=[32], std=True, C=0.05),
    'wl1_31': dict(model='whisper_large_v1', layers=[31], std=True, C=0.05),
    'him22': dict(model='whisper_hi_medium', layers=[22], std=True, C=0.05),
    'hil2_31': dict(model='whisper_hi_large_v2', layers=[31], std=True, C=0.05),
}
P = f'{BH}/work/probs/members.pkl'


def blend_probs(R, keys, part=0):
    return np.exp(np.mean([np.log(R[k][part] + 1e-9) for k in keys], 0))


if __name__ == '__main__':
    m = load_meta()
    R = pickle.load(open(P, 'rb')) if os.path.exists(P) else {}
    for k, s in MEMBERS.items():
        if k not in R:
            R[k] = blend.run_spec(m, s); pickle.dump(R, open(P, 'wb'))
        print(k, blend.evaluate(m, R[k][0]).round(4), flush=True)
    sel, best = [], -1
    for step in range(10):
        cand = [(blend.evaluate(m, blend_probs(R, sel + [k]))[0], k) for k in R]
        sc, k = max(cand)
        if sc <= best + 1e-4:
            break
        sel.append(k); best = sc
        print('step', step, sel, blend.evaluate(m, blend_probs(R, sel)).round(4), flush=True)
