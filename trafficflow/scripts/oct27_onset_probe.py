"""2026-10-27 onset probes on the L1 line: combine the incumbent onset rows (public kernel T08 + O1 safe v8 cells) with
our v8 onset probabilities (probs_v8_seeds9_stack03_onset: official windows, T+30 only, data <= T).

    python oct27_onset_probe.py table                  # per-window overlap stats
    python oct27_onset_probe.py build MODE PARAM OUT   # MODE: union (add v8 p >= PARAM), prune (drop incumbent cells
                                                       # with v8 p < PARAM), blend (eIoU top-m on PARAM*inc + (1-PARAM)*p)
Only the T+30 onset cells change (steps 1-5 stay empty); ongoing, state and ODME rows stay byte-identical to the base.
"""
import sys

import numpy as np
import pandas as pd

from trafficflow.data import tindex
from trafficflow.t2.core import PANELS8, official_windows, statics
from trafficflow.t2.models import eiou_topm

BASE = "/home/urad/tfbroot/work/subs/L1_state.csv"
KEY = "/home/urad/tfbroot/work/tfbw/work__pub__submission_key.parquet"
PROBS = "/home/urad/tfbroot/work/t2h/work__t2h__probs_v8_seeds9_stack03_onset.parquet"


def load():
    sub = pd.read_csv(BASE, dtype=str)
    key = pd.read_parquet(KEY, columns=["submission_id", "window_id", "timestamp", "link_id"])
    key["submission_id"] = key.submission_id.astype(str)
    sub = sub.merge(key, on="submission_id", how="left", sort=False)
    assert len(sub) == 6980503
    W = pd.concat([official_windows(p, s) for p in PANELS8 for s in ("validation", "private")]).set_index("window_id")
    on = np.flatnonzero(sub.window_id.map(W.condition).to_numpy() == "queue_onset")
    s = sub.iloc[on]
    k = tindex(s.timestamp) - s.window_id.map(W["T"]).to_numpy()      # 1..6 (T+5 .. T+30)
    lid = np.array([statics(W.loc[w, "panel"])["lid"][l] for w, l in zip(s.window_id, s.link_id)])
    R = pd.DataFrame({"row": on, "window_id": s.window_id.values, "k": k, "link": lid,
                      "inc": s.queue_pred.astype(int).values})
    P = pd.read_parquet(PROBS)
    P["k"] = P.k.astype(int); P["link"] = P.link.astype(int)
    R = R.merge(P[["window_id", "k", "link", "p", "split"]], on=["window_id", "k", "link"], how="left")
    return sub, R


def table(R):
    t = R[R.k == 6]
    print("incumbent cells off T+30:", int(R[R.k != 6].inc.sum()), "| missing p at T+30:", int(t.p.isna().sum()))
    for sp, g in t.groupby("split"):
        print(sp, "windows", g.window_id.nunique(), "inc cells", int(g.inc.sum()),
              "| p>=0.5 cells", int((g.p >= 0.5).sum()), "of which inc", int(((g.p >= 0.5) & (g.inc == 1)).sum()))
        print("  inc cells by v8 p bin:", pd.cut(g.p[g.inc == 1], [0, .02, .05, .1, .2, .3, .5, 1]).value_counts().sort_index().to_dict())
        print("  non-inc cells by v8 p bin:", pd.cut(g.p[g.inc == 0], [.2, .3, .4, .5, .6, .8, 1]).value_counts().sort_index().to_dict())


def build(sub, R, mode, x, out):
    t = R[R.k == 6].copy()
    new = t.inc.copy()
    if mode == "union":
        new = ((t.inc == 1) | (t.p >= x)).astype(int)
    elif mode == "prune":
        new = ((t.inc == 1) & (t.p >= x)).astype(int)
        for _, g in t[t.inc == 1].groupby("window_id"):       # never empty a window: keep its most likely cell
            if new.loc[g.index].sum() == 0:
                new.loc[g.p.idxmax()] = 1
    elif mode == "blend":
        t["q"] = x * t.inc + (1 - x) * t.p
        new = pd.Series(0, index=t.index)
        for _, g in t.groupby("window_id"):
            idx, _ = eiou_topm(g.q.to_numpy())
            new.loc[g.index[idx]] = 1
    else:
        raise SystemExit(mode)
    t["new"] = new
    for sp, g in t.groupby("split"):
        print(sp, "added", int(((g.new == 1) & (g.inc == 0)).sum()), "removed", int(((g.new == 0) & (g.inc == 1)).sum()),
              "windows changed", int(g.groupby("window_id").apply(lambda h: (h.new != h.inc).any()).sum()),
              "empty windows", int((g.groupby("window_id").new.sum() == 0).sum()))
    q = sub.queue_pred.copy()
    q.iloc[t.row.to_numpy()] = t.new.astype(str).to_numpy()
    sub = sub.assign(queue_pred=q)[["submission_id", "task", "speed_kmh", "flow_vph", "queue_pred", "path_flow"]]
    sub.to_csv(out, index=False)
    print("wrote", out)


if __name__ == "__main__":
    sub, R = load()
    if sys.argv[1] == "table":
        table(R)
    else:
        build(sub, R, sys.argv[2], float(sys.argv[3]), sys.argv[4])
