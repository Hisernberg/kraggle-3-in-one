"""Variants of the month-specific speed correction (t1_bias.py), cross-fitted on the Task 1 pseudo-holdout.

2026-10-09: none beats the adopted scheme (per panel, link, month, raw-speed band; K = 30). In S_total units
(0.35 x dS), vs the adopted scheme, March / April: K=10 -0.00002 / -0.00000; K=100 -0.00002 / -0.00003;
+ time-of-day period -0.00004 / -0.00006; + regime -0.00005 / -0.00005; no band -0.00005 / -0.00007.

    python -m trafficflow.t1_bias_variants      # needs WORK/pseudo/<panel>.parquet and pred_fullP2/fullP3 (t1_pseudo)
"""
import numpy as np, pandas as pd
from . import t1_bias as B
from .data import SLOTS
from .t1_pseudo import s_state

X = B.pseudo_frame(B._members("fullP2:0.5,fullP3:0.5"))
h = (X.t % SLOTS) // 12
X["tod"] = np.digitize(h, [6, 10, 15, 19])
X["reg"] = X.regime_day
fold = (X.t // SLOTS) % 2

def run(keys, K):
    c = np.zeros(len(X))
    for f in (0, 1):
        tr = X[fold != f]
        G = tr.groupby(keys).agg(n=("r", "size"), c=("r", "mean")).reset_index()
        G["c"] *= G.n / (G.n + K)
        c[(fold == f).to_numpy()] = X.loc[fold == f, keys].merge(G, on=keys, how="left").c.fillna(0).to_numpy()
    out = {}
    for (panel, split), g in X.groupby(["panel", "split"]):
        i = g.index.to_numpy(); e = np.exp(c[i])
        a = (g.y_speed.to_numpy(), g.y_flow.to_numpy(), g.regime_day.to_numpy())
        out[(panel, split)] = s_state(g.v.to_numpy() * e, g.q.to_numpy() * e, *a) - s_state(g.v.to_numpy(), g.q.to_numpy(), *a)
    s = pd.Series(out).unstack()
    return s

base = None
for name, keys, K in [("base", B.KEYS, 30), ("K10", B.KEYS, 10), ("K100", B.KEYS, 100),
                      ("tod", B.KEYS + ["tod"], 30), ("tod_K60", B.KEYS + ["tod"], 60),
                      ("reg", B.KEYS + ["reg"], 30), ("noband", ["panel", "j", "split"], 30)]:
    s = run(keys, K)
    if base is None: base = s
    d = s - base
    print(f"{name:8s} 0.35*dS Mar {0.35*s.validation.mean():.6f} Apr {0.35*s.private.mean():.6f} | vs base Mar {0.35*d.validation.mean():+.6f} ({(d.validation>0).sum()}/10) Apr {0.35*d.private.mean():+.6f} ({(d.private>0).sum()}/10)", flush=True)
