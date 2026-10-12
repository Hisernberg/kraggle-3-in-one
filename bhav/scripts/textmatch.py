"""Transcript near-duplicate override: a clip whose (Hindi Whisper-small) transcript has character similarity >= TH
with train clips from other duplicate clusters takes their similarity-weighted majority label. Applied after the
waveform-duplicate override. CV: same 3 x 5 folds; only rows without a waveform duplicate in the training folds."""
import sys, os
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from common import *
import blend

S = np.load(f'{BH}/work/text_sim.npy')


def evaluate_tm(m, oof, th):
    tr = (m.split == 'train').values; ti = np.where(tr)[0]; y = m.y.values[tr].astype(int); cl = m.cl.values[tr]
    Str = S[np.ix_(ti, ti)]; out = []
    for si, s in enumerate(blend.SEEDS):
        f = folds(y, s); pred = oof[si].argmax(1).copy()
        from probe import novel_mask
        nov = novel_mask(cl, f)
        for i in range(len(y)):
            same = (cl == cl[i]) & (f != f[i])
            if same.any():
                pred[i] = np.bincount(y[same], minlength=4).argmax(); continue
            cand = (Str[i] >= th) & (f != f[i]) & (cl != cl[i])
            if cand.any():
                pred[i] = np.bincount(y[cand], weights=Str[i][cand], minlength=4).argmax()
        out.append((mf1(y[nov], pred[nov]), mf1(y, pred)))
    return np.array(out).mean(0)


def finalize_tm(m, test_prob, th):
    t = blend.finalize(m, test_prob)
    tr = (m.split == 'train').values; ti = np.where(tr)[0]; tei = np.where(~tr)[0]
    y = m.y.values[tr].astype(int); cl = m.cl.values[tr]; changed = []
    for k, j in enumerate(tei):
        if t.src[k] == 'dup':
            continue
        cand = (S[j, ti] >= th) & (cl != m.cl.values[j])
        if cand.any():
            lab = CLASSES[np.bincount(y[cand], weights=S[j, ti][cand], minlength=4).argmax()]
            if lab != t.emotion[k]:
                changed.append((int(t.Id[k]), t.emotion[k], lab, round(float(S[j, ti][cand].max()), 3)))
            t.loc[k, 'emotion'] = lab; t.loc[k, 'src'] = 'text'
    return t, changed
