"""Shared helpers: data loading, duplicate clusters, CV folds, metrics."""
import os
import numpy as np, pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

BH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = f'{BH}/work/data/SRCASW-BhavVaani'
EMB = f'{BH}/work/emb'
CLASSES = ['angry', 'happy', 'neutral', 'sad']


def load_meta():
    """Train/test rows in the order used by the extraction kernel, plus duplicate cluster ids."""
    tr = pd.read_csv(f'{DATA}/train.csv'); te = pd.read_csv(f'{DATA}/test.csv')
    m = pd.concat([tr.assign(split='train'), te.assign(split='test')], ignore_index=True)
    cl = pd.read_csv(f'{BH}/work/clusters.csv')          # Id -> cluster id (from dupgraph.py)
    m['cl'] = m.Id.map(dict(zip(cl.Id, cl.cl)))
    m['y'] = m.emotion.map({c: i for i, c in enumerate(CLASSES)})
    return m


def folds(y, seed, k=5):
    f = np.zeros(len(y), int)
    for i, (_, va) in enumerate(StratifiedKFold(k, shuffle=True, random_state=seed).split(np.zeros(len(y)), y)):
        f[va] = i
    return f


def mate_probs(cl_tr, y_tr, cl_q, exclude_self=None):
    """Label distribution of train rows sharing each query's duplicate cluster (None when none)."""
    out = []
    for qi, c in enumerate(cl_q):
        mask = cl_tr == c
        if exclude_self is not None:
            mask = mask & exclude_self[qi]
        if mask.any():
            out.append(np.bincount(y_tr[mask], minlength=4) / mask.sum())
        else:
            out.append(None)
    return out


def mf1(y, p):
    return f1_score(y, p, average='macro')
