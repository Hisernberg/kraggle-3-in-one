"""Score eligibility in Task 2 (official scorer: IoU over is_score_eligible cells only).

`score_task2.score_window` keeps only the target rows with is_score_eligible = True, so a prediction
or a truth cell on an ineligible cell counts for nothing. Our truth, CV and decoder count every cell.
This module measures the gap and tests an eligibility-aware expected-IoU decoder:

    maximise sum_S p_i e_i / (sum_S e_i + sum_notS p_i e_i)

where e_i is the probability that cell i is eligible. On train the horizon eligibility is known
(oracle). At test time it has to be estimated from the window history (data <= T).

    T2_WORK=/home/user/work/t2h T2_FEAT=/home/user/work/t2h/feat python -m trafficflow.t2.elig onset
    T2_WORK=/home/user/work/t2  T2_FEAT=/home/user/work/t2/feat_v3 python -m trafficflow.t2.elig ongoing
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from ..data import load
from .core import K, PANELS8, WORK, aggregate, iou
from .models import eiou_topm
from .robust import meta_index


def eiou_weighted(p: np.ndarray, e: np.ndarray) -> np.ndarray:
    """Indices of the set maximising sum_S p e / (sum_S e + sum_notS p e), taken over a top-m by p.
    Cells with e = 0 never change the objective; they are left out."""
    o = np.argsort(-p)
    pe = (p * e)[o]; ee = e[o]
    cpe = np.cumsum(pe); ce = np.cumsum(ee); tot = pe.sum()
    r = cpe / np.maximum(ce + tot - cpe, 1e-12)
    j = int(np.argmax(r))
    sel = o[: j + 1]
    return sel[e[sel] > 0] if (e[sel] > 0).any() else sel


def decode_empty(p: np.ndarray, e: np.ndarray, base: np.ndarray | None = None, margin: float = 0.0) -> np.ndarray:
    """Eligibility-aware decoding with an explicit "nothing eligible" option.

    The official IoU is 1.0 when neither the truth nor the prediction has an eligible cell. Choose it
    when P(no eligible truth cell) = prod(1 - p e) exceeds the surrogate of the best non-empty
    eligible set (+ margin). Then keep only the cells of `base` (default top-m) with e < 0.5: they are
    free if ineligible and a hedge if they turn out eligible."""
    S = eiou_weighted(p, e)
    pe = p * e
    rS = pe[S].sum() / max(e[S].sum() + pe.sum() - pe[S].sum(), 1e-12)
    p_empty = float(np.prod(1.0 - np.clip(pe, 0, 1)))
    if p_empty > rS + margin:
        b = eiou_topm(p)[0] if base is None else base
        return b[e[b] < 0.5]
    return S


def _elig_arrays():
    out = {}
    for p in PANELS8:
        out[p] = load(p)["elig"]
    return out


def evaluate(cond: str, O: pd.DataFrame, truth_key: str = "y") -> pd.DataFrame:
    """O: rows gw, k, link, p (OOF probabilities of the evaluated model). Returns per-window IoU for
    decoders x scoring (all cells = our CV; eligible only = official)."""
    Mi = meta_index(cond)
    O = O[O.gw.map(Mi.src).isin(["sim", "off"]).to_numpy()].sort_values("gw", kind="stable")
    EL = _elig_arrays()
    DS = {p: np.load(WORK / f"ds_{p}.npz", allow_pickle=True) for p in PANELS8}
    g = O.gw.to_numpy(); cut = np.flatnonzero(np.diff(g)) + 1
    kk = O.k.to_numpy().astype(int) - 1; ll = O.link.to_numpy().astype(int); pp = O.p.to_numpy()
    rows = []
    for idx in np.split(np.arange(len(O)), cut):
        gw = int(g[idx[0]]); m = Mi.loc[gw]; p = m.panel; w = int(m.w)
        z = DS[p]
        T = int(z["w_T"][w])
        yt = z[truth_key][w].astype(bool).copy()
        if cond == "queue_onset":
            yt[:K - 1] = False
        E = EL[p][T + 1:T + K + 1].astype(bool)                      # horizon eligibility [K, L] (oracle, train only)
        he = z["he"][w].astype(np.float32)                           # history eligibility [H, L] (<= T)
        e_hat = {"last": he[-1], "mean": he.mean(0), "any": (he.max(0) > 0).astype(np.float32)}
        ki, li, pr = kk[idx], ll[idx], pp[idx]
        decs = {"topm": eiou_topm(pr)[0],
                "oracle": eiou_weighted(pr, E[ki, li].astype(np.float64))}
        decs["oracle_empty"] = decode_empty(pr, E[ki, li].astype(np.float64))
        for nm, eh in e_hat.items():
            decs[f"w_{nm}"] = eiou_weighted(pr, eh[li].astype(np.float64))
            decs[f"empty_{nm}"] = decode_empty(pr, eh[li].astype(np.float64))
        for dn, sel in decs.items():
            P = np.zeros_like(yt); P[ki[sel], li[sel]] = True
            rows.append(dict(gw=gw, panel=p, condition=cond, src=m.src, dec=dn,
                             iou_all=iou(P, yt), iou_elig=iou(P & E, yt & E),
                             ntrue=int(yt.sum()), ntrue_elig=int((yt & E).sum()), npred=int(P.sum()),
                             npred_elig=int((P & E).sum())))
    return pd.DataFrame(rows)


def summary(df: pd.DataFrame, cond: str) -> pd.DataFrame:
    res = []
    for dn, d in df.groupby("dec", sort=False):
        s = d[d.src == "sim"]; o = d[d.src == "off"]
        res.append(dict(dec=dn, sim_all=aggregate(s, "iou_all")[cond], sim_elig=aggregate(s, "iou_elig")[cond],
                        off_all=aggregate(o, "iou_all")[cond], off_elig=aggregate(o, "iou_elig")[cond],
                        ntrue=s.ntrue.mean(), ntrue_elig=s.ntrue_elig.mean(), npred=s.npred.mean(),
                        npred_elig=s.npred_elig.mean()))
    return pd.DataFrame(res).round(4)


if __name__ == "__main__":
    cond = {"onset": "queue_onset", "ongoing": "queue_ongoing"}[sys.argv[1]]
    if cond == "queue_onset":
        from .calib import load_oof
        O = load_oof()
    else:
        from .calib import V5_ONGOING
        from .v5_eval import blend_oof
        O = blend_oof(V5_ONGOING)
    df = evaluate(cond, O, truth_key="y")
    df.to_parquet(WORK / f"elig_{sys.argv[1]}_windows.parquet")
    print(summary(df, cond).to_string(index=False))
