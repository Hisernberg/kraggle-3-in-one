"""Recurrence-prior features for the T08-style onset rows (data <= T: train-split truth only).

prior(l, s) = share of train days on which link l is queued at time-of-day slot s (+-2 slots), split by
weekday / weekend. Train rows use the leave-own-day-out share; official windows use all train days.

    python -m trafficflow.t2.onset_prior rows      # -> t08/rows_prior.npz (aligned with rows.npz)
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from ..data import SLOTS, SPLIT_DAYS, T0, load
from .core import statics
from .t08_features import HORIZON, T08Panel, panel_names

W08 = "/home/urad/tfbroot/work/t08"
NP = 4


def tables(name: str, P: T08Panel):
    st = statics(name); perm = np.array([st["lid"][l] for l in P.links])
    d = load(name)
    a, b = SPLIT_DAYS["train"]
    v = d["tspeed"][a * SLOTS:b * SLOTS, perm].reshape(b - a, SLOTS, -1)
    q = ((v <= P.cut[None, None]) & np.isfinite(v)).astype(np.float32)
    n = np.isfinite(v).astype(np.float32)
    k = np.ones(5, np.float32)
    sm = lambda x: np.apply_along_axis(lambda c: np.convolve(np.pad(c, 2, mode="edge"), k, "valid"), 1, x)
    qs, ns = sm(q), sm(n)                                   # [days, slots, L] counts in +-2 slots
    wk = np.array([(T0 + pd.Timedelta(days=int(dd))).dayofweek >= 5 for dd in range(a, b)])
    return qs, ns, wk


def feats(qs, ns, wk, day, t, link, loo: bool):
    """[n, NP]: prior at T+30 and at T (same day type), and at T+30 / T over all days."""
    out = []
    dt = (T0 + pd.Timedelta(days=int(day))).dayofweek >= 5
    for slot in (t + HORIZON, t):
        slot = min(slot, SLOTS - 1)
        for same in (True, False):
            sel = (wk == dt) if same else np.ones(len(wk), bool)
            Q = qs[sel, slot, :].sum(0); N = ns[sel, slot, :].sum(0)
            if loo and day < len(wk) and sel[day]:
                Q = Q - qs[day, slot, :]; N = N - ns[day, slot, :]
            out.append((Q / np.maximum(N, 1))[link])
    return np.column_stack(out).astype(np.float32)


def rows():
    z = np.load(f"{W08}/rows.npz"); X, meta = z["X"], z["meta"]
    out = np.zeros((len(X), NP), np.float32)
    for pi, name in enumerate(panel_names()):
        P = T08Panel(name, pi); qs, ns, wk = tables(name, P)
        sel = np.flatnonzero(meta[:, 0] == pi)
        dfr = pd.DataFrame({"i": sel, "day": meta[sel, 1], "t": meta[sel, 3], "link": X[sel, 1].astype(int)})
        for (day, t), g in dfr.groupby(["day", "t"]):
            out[g.i.to_numpy()] = feats(qs, ns, wk, int(day), int(t), g.link.to_numpy(), True)
        print(name, flush=True)
    np.savez_compressed(f"{W08}/rows_prior.npz", Xr=out)


if __name__ == "__main__":
    if sys.argv[1] == "rows":
        rows()


def for_windows(rows) -> np.ndarray:
    """Prior features for onset_t08.official / pseudo rows (all train days, no leave-out), stacked like their X."""
    cache, out = {}, []
    names = panel_names()
    for r in rows:
        name, origin, links = r[1], r[3], r[4]
        if name not in cache:
            cache[name] = tables(name, T08Panel(name, names.index(name)))
        qs, ns, wk = cache[name]
        mins = int((origin - T0).total_seconds() // 60)
        day, t = mins // (5 * SLOTS), (mins // 5) % SLOTS
        out.append(feats(qs, ns, wk, day, t, np.arange(len(links)), False))
    return np.concatenate(out)
