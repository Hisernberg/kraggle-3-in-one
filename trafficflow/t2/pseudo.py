"""Task 2 test-month pseudo-holdout: queue windows found in the observed March/April masked view.

Plain CV draws its windows from train, which shares one demand draw; March and April are independent draws, and
4 of the last 5 ongoing changes that won in CV lost on March. This module builds windows from the test months
themselves, from released data only, so a change can be checked on the months that are scored.

Origins T (every slot of the split) are kept when, on the masked view:
- no corridor-dark row (the official windows' horizons and buffers) lies in [T-12, T+6];
- the history [T-12, T) has eligible coverage >= 0.7 (the official selector's rule);
- the condition is read from the history after a causal fill of the hidden Task 1 target cells (last value of the
  link, at most 3 slots back; official histories carry no targets): ongoing if >= 2 queued eligible observations with
  >= 2 on one link, else onset;
- at least one observed, eligible horizon cell (T+1..T+6) is queued;
- ongoing: persistence IoU of slot T (filled) against the observed horizon <= 0.9; every 6th candidate is kept;
- onset: only the first origin of a run of onset candidates (the chronological selector draws only those).
Labels are the queue status of the observed eligible horizon cells; hidden cells (Task 1 targets, missing values) are
unknown and left out of the IoU, which is then an unbiased estimate of the IoU on eligible cells.
Features: ``build_features._rows`` on the filled history, the masked view (slot T, earlier hours, previous 7 days)
and the full-train profile, as for the official windows.

    T2_WORK=/home/user/work/t2 T2_FEAT=/home/user/work/t2/feat_v3 T2_PHYSICS=1 \\
        python -m trafficflow.t2.pseudo build            # WORK/pseudo2/feat_<panel>.parquet, windows_<panel>.npz
    ... python -m trafficflow.t2.pseudo score queue_ongoing '{"v5": [["lgb_v5_og_v3", 0.35], ...], ...}'
"""
from __future__ import annotations

import gc
import json
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from ..data import SLOTS, SPLIT_DAYS, load
from .build_features import _rows, masked_view
from .core import H, K, PANELS8, WORK, aggregate, statics
from .dataset import load_ds
from .features import profiles
from .robust_pipeline import decode_topm

OUT = WORK / "pseudo2"
OG_EVERY = 6
FILL = 3


def _ffill_targets(x: np.ndarray, hidden: np.ndarray, limit: int = FILL) -> np.ndarray:
    """Fill hidden cells with the last observed value of the same link, at most ``limit`` slots back (causal)."""
    f = x.copy()
    for k in range(1, limit + 1):
        m = hidden & np.isnan(f)
        m[:k] = False
        src = np.roll(x, k, 0)
        f[m] = src[m]
    return f


def find_windows(panel: str, split: str):
    st = statics(panel)
    vcut = st["vcut"].astype(np.float32)
    d = load(panel)
    a, b = SPLIT_DAYS[split]
    lo, hi = a * SLOTS, b * SLOTS
    sp, fl, el, pct = d["speed"], d["flow"], d["elig"], d["pct"]
    tg = d["target"] > 0
    hidden = tg & np.isnan(sp)
    spf = _ffill_targets(sp, hidden)
    flf = _ffill_targets(fl, hidden)
    dark = np.isnan(sp).all(1)
    known = el & ~np.isnan(sp)
    qk = known & (sp <= vcut[None, :])
    qf = el & ~np.isnan(spf) & (spf <= vcut[None, :])
    cand = []
    for T in range(lo + H, hi - K):
        if dark[T - H:T + K + 1].any():
            continue
        if el[T - H:T].mean() < 0.7:
            continue
        yk = qk[T + 1:T + K + 1]
        if not yk.any():
            continue
        per = qf[T - H:T].sum(0)
        og = per.sum() >= 2 and per.max() >= 2
        if og:
            kn = known[T + 1:T + K + 1]
            pers = np.repeat(qf[T][None, :], K, 0)
            u = ((pers | yk) & kn).sum()
            if u and ((pers & yk) & kn).sum() / u > 0.9:
                continue
        cand.append((T, og))
    cand = pd.DataFrame(cand, columns=["T", "og"])
    first_on = (~cand.og & ~(cand["T"] - 1).isin(cand["T"])).to_numpy()
    keep_og = cand.og.to_numpy() & (cand["T"].to_numpy() % OG_EVERY == 0)
    W = cand[first_on | keep_og].reset_index(drop=True)
    W["condition"] = np.where(W.og, "queue_ongoing", "queue_onset")
    Ts = W["T"].to_numpy()
    hidx = Ts[:, None] + np.arange(-H, 0)[None, :]
    fidx = Ts[:, None] + np.arange(1, K + 1)[None, :]
    arr = dict(hs=spf[hidx], hf=flf[hidx], he=el[hidx], hp=pct[hidx].astype(np.int8), y=qk[fidx], known=known[fidx])
    return W[["T", "condition"]], arr


def build(panels=PANELS8) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for panel in panels:
        t0 = time.time()
        _, A = load_ds(panel)
        prof = profiles(A["qtrue"], None, np.ones(A["qtrue"].shape[0] // SLOTS, bool))
        del A
        mv = masked_view(panel)
        feats, metas, arrs = [], [], {}
        base = 0
        for split in ("validation", "private"):
            W, arr = find_windows(panel, split)
            W["split"] = split
            W.index = np.arange(base, base + len(W)); base += len(W)
            W["window_id"] = [f"PS_{panel}_{i}" for i in W.index]
            for cond in ("queue_onset", "queue_ongoing"):
                sel = np.flatnonzero((W.condition == cond).to_numpy())
                for c0 in range(0, len(sel), 150):
                    s = sel[c0:c0 + 150]
                    X = _rows(panel, arr["hs"][s], arr["hf"][s], arr["he"][s], arr["hp"][s], W["T"].to_numpy()[s], cond,
                              prof, mv=mv, wkey=W.index.to_numpy()[s])
                    feats.append(X)
            metas.append(W)
            for k, v in arr.items():
                arrs.setdefault(k, []).append(v)
        X = pd.concat(feats, ignore_index=True)
        X["panel"] = panel
        M = pd.concat(metas)
        M.index.name = "w"
        f64 = X.select_dtypes("float64").columns
        X[f64] = X[f64].astype(np.float32)
        X.to_parquet(OUT / f"feat_{panel}.parquet")
        M.reset_index().to_parquet(OUT / f"meta_{panel}.parquet")
        np.savez_compressed(OUT / f"windows_{panel}.npz", **{k: np.concatenate(v) for k, v in arrs.items()})
        print(panel, M.groupby(["split", "condition"]).size().to_dict(), X.shape, f"{time.time() - t0:.0f}s", flush=True)


def load_cond(cond: str, p: str):
    """Feature rows, window meta and (labels, known) of one panel and condition."""
    X = pd.read_parquet(OUT / f"feat_{p}.parquet")
    M = pd.read_parquet(OUT / f"meta_{p}.parquet").set_index("w")
    X["pcode"] = np.float32(PANELS8.index(p))
    X["window_id"] = X.w.map(M.window_id)
    X = X[X.w.map(M.condition) == cond].reset_index(drop=True)
    if cond == "queue_onset":            # onset-prior columns, as pipeline.load_split adds them
        from .oprior import COLS as OP_COLS, OnsetPrior
        ws = np.unique(X.w)
        f = OnsetPrior(p).features(M.loc[ws, "T"].to_numpy(), None)
        pos = {w: i for i, w in enumerate(ws)}
        wi = X.w.map(pos).to_numpy(); li = X.link.to_numpy().astype(int)
        for c in OP_COLS:
            X[c] = f[c][wi, li]
    z = np.load(OUT / f"windows_{p}.npz")
    Ys = {M.loc[w, "window_id"]: (z["y"][w], z["known"][w]) for w in M.index[M.condition == cond]}
    return X, M.assign(panel=p), Ys


def window_iou(pred: dict, Ys: dict, M: pd.DataFrame) -> pd.DataFrame:
    rows = []
    Mi = M.set_index("window_id")
    for wid, (y, kn) in Ys.items():
        A = pred.get(wid, np.zeros_like(y))
        u = ((A | y) & kn).sum()
        rows.append(dict(window_id=wid, panel=Mi.loc[wid, "panel"], condition=Mi.loc[wid, "condition"],
                         split=Mi.loc[wid, "split"], iou=1.0 if u == 0 else float(((A & y) & kn).sum() / u),
                         n_true=int((y & kn).sum()), n_pred=int(A.sum())))
    return pd.DataFrame(rows)


def _booster(m: str, cond: str) -> lgb.Booster:
    """"<name>" -> WORK/model_<name>_<cond>.txt; "t2h:<name>" -> /home/user/work/t2h/model_<name>_<cond>.txt."""
    d, n = (m.split(":", 1) if ":" in m else (str(WORK), m))
    d = {"t2h": "/home/user/work/t2h"}.get(d, d)
    return lgb.Booster(model_file=f"{d}/model_{n}_{cond}.txt")


def score(cond: str, schemes: dict, panels=PANELS8) -> pd.DataFrame:
    """schemes: name -> [(saved model name, weight), ...]; probabilities are the weighted mean (as robust_pipeline).
    One panel at a time (the ongoing tables of all panels do not fit in memory next to their copies)."""
    names = sorted({m for v in schemes.values() for m, _ in v})
    B = {m: _booster(m, cond) for m in names}
    out = []
    for p in panels:
        X, M, Ys = load_cond(cond, p)
        P = {m: b.predict(X[b.feature_name()].to_numpy(np.float32), num_threads=4) for m, b in B.items()}
        key = X[["window_id", "panel", "k", "link"]].copy()
        del X
        gc.collect()
        for name, members in schemes.items():
            pr = sum(w * P[m] for m, w in members) / sum(w for _, w in members)
            out.append(window_iou(decode_topm(key.assign(p=pr)), Ys, M).assign(scheme=name))
        print(p, "scored", flush=True)
    d = pd.concat(out, ignore_index=True)
    d.to_csv(OUT / f"score_{cond}.csv", index=False)
    summarize(d, list(schemes))
    return d


def summarize(d: pd.DataFrame, order: list) -> pd.DataFrame:
    """Official aggregation per month; dS vs the first scheme with a paired window bootstrap SE."""
    rows = []
    ref = order[0]
    rng = np.random.default_rng(0)
    for split in ("validation", "private"):
        g = d[d.split == split]
        base = g[g.scheme == ref].set_index("window_id")
        for name in order:
            h = g[g.scheme == name].set_index("window_id")
            S = aggregate(h.reset_index())["overall"]
            diff = (h.iou - base.iou.reindex(h.index)).to_numpy()
            se = np.std([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(200)])
            rows.append(dict(scheme=name, split=split, n=len(h), S=S, dS=S - aggregate(base.reset_index())["overall"],
                             mean_d=diff.mean(), se_mean_d=se, better=int((diff > 1e-9).sum()), worse=int((diff < -1e-9).sum())))
    r = pd.DataFrame(rows)
    print(r.round(4).to_string(index=False))
    return r


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "build":
        build(sys.argv[2:] or PANELS8)
    elif cmd == "score":
        spec = json.loads(sys.argv[3])
        score(sys.argv[2], {k: [(m, float(w)) for m, w in v] for k, v in spec.items()})
