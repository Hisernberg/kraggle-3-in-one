"""Retrained T08-style onset models, scored on the pseudo onset windows against the released T08 models.

    python -m trafficflow.t2.onset_t08_train rows               # train rows (T08 recipe: first origin of every onset
                                                                #  candidate run on full-truth train days) -> t08/rows.npz
    python -m trafficflow.t2.onset_t08_train fit NAME [opts]    # LightGBM; opts: patch=1 drop=0.05 adapt=W seed=S
    python -m trafficflow.t2.onset_t08_train score NAME[,NAME]  # pseudo3 / pseudo2 window IoU vs T08

Data <= T: train-split truth only; ``adapt=W`` adds the March pseudo onset windows (released masked view, observed
labels) with weight W, which is legal for April windows only (all of March is <= T for every April window).
"""
from __future__ import annotations

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from ..data import SLOTS, SPLIT_DAYS, T0, load
from .core import statics
from .t08_features import HISTORY, HORIZON, T08Panel, panel_names

W08 = Path("/home/urad/tfbroot/work/t08")
PARAMS = {"objective": "binary", "learning_rate": .04, "num_leaves": 15, "min_data_in_leaf": 100, "lambda_l2": 10,
          "feature_fraction": .9, "num_threads": 4, "verbosity": -1, "seed": 20261004}


def patch(P: T08Panel, speed: np.ndarray) -> np.ndarray:
    """[L, 45] raw space-time patch: speed ratio at lags 1..5 for milepost neighbours -4..4."""
    s = speed.astype(np.float32) / P.vf[None]
    cols = []
    for lag in range(1, 6):
        for off in range(-4, 5):
            cols.append(s[-lag] if off == 0 else P._mpn(s[-lag], off))
    X = np.column_stack(cols).astype(np.float32)
    X[~np.isfinite(X)] = np.nan
    return X


def candidates(P, v, e):
    """T08 candidate_origins on one day (onset only), first origin of each run."""
    qd = (v <= P.cut[None]) & e & np.isfinite(v)
    act = qd.any(1)
    out = []
    for t in range(HISTORY, len(v) - HORIZON):
        if np.isfinite(v[t - HISTORY:t]).mean() < .7 or not act[t + 1:t + HORIZON + 1].any():
            continue
        if not act[t - HISTORY:t].any() and not act[t]:
            out.append(t)
    s = set(out)
    return [t for t in out if t - 1 not in s]


def build_rows(drop: float = 0.05, reps: int = 2):
    rng = np.random.default_rng(20261005)
    X, Xp, y, meta = [], [], [], []
    for pi, name in enumerate(panel_names()):
        P = T08Panel(name, pi); st = statics(name)
        perm = np.array([st["lid"][l] for l in P.links])
        d = load(name)
        V, Q, E = d["tspeed"][:, perm], d["tflow"][:, perm], d["elig"][:, perm]
        a, b = SPLIT_DAYS["train"]
        for day in range(a, b):
            sl = slice(day * SLOTS, (day + 1) * SLOTS)
            v, q, e = V[sl], Q[sl], E[sl]
            if np.isnan(v).all():
                continue
            for t in candidates(P, v, e):
                lab = (v[t + HORIZON] <= P.cut) & np.isfinite(v[t + HORIZON])
                ok = e[t + HORIZON] & np.isfinite(v[t + HORIZON])
                origin = T0 + pd.Timedelta(minutes=5 * (day * SLOTS + t))
                for rep in range(reps):
                    hv, hq, he = v[t - HISTORY:t].copy(), q[t - HISTORY:t].copy(), e[t - HISTORY:t].astype(np.float32)
                    if rep:
                        m = rng.random(hv.shape) < drop
                        hv[m] = np.nan; hq[m] = np.nan; he[m] = 0
                    X.append(P.onset_rows(hv, hq, he, origin)[ok]); Xp.append(patch(P, hv)[ok]); y.append(lab[ok])
                    meta.append(np.column_stack([np.full(ok.sum(), pi), np.full(ok.sum(), day), np.full(ok.sum(), rep), np.full(ok.sum(), t)]))
        print(name, "rows", sum(len(x) for x in y), flush=True)
    np.savez_compressed(W08 / "rows.npz", X=np.concatenate(X), Xp=np.concatenate(Xp), y=np.concatenate(y),
                        meta=np.concatenate(meta))


def pseudo_rows(d: str):
    """Feature rows + labels of the pseudo onset windows (from onset_t08 pseudo output plus the patch block)."""
    from .onset_t08 import pseudo as prow
    import os
    os.environ["T08_PSEUDO"] = f"/home/urad/tfbroot/work/t2/{d}"
    import trafficflow.t2.onset_t08 as ot
    ot.PSEUDO = Path(f"/home/urad/tfbroot/work/t2/{d}")
    rows = prow(None)
    PS = {n: T08Panel(n, i) for i, n in enumerate(panel_names())}
    out = []
    for split, name, wid, origin, links, X, y, kn in rows:
        st = statics(name); perm = np.array([st["lid"][l] for l in links])
        M = pd.read_parquet(ot.PSEUDO / f"meta_{name}.parquet")
        w = int(M.index[M.window_id == wid][0]) if "w" not in M.columns else int(M.loc[M.window_id == wid, "w"].iloc[0])
        z = np.load(ot.PSEUDO / f"windows_{name}.npz")
        Xp = patch(PS[name], z["hs"][w][:, perm])
        out.append((split, name, wid, links, X, Xp, y[-1], kn[-1]))
    return out


_PR = {}


def get_pseudo(d):
    if d not in _PR:
        f = W08 / f"prow_{d}.pkl"
        if f.exists():
            _PR[d] = pd.read_pickle(f)
        else:
            _PR[d] = pseudo_rows(d); pd.to_pickle(_PR[d], f)
    return _PR[d]


def fit(name: str, patch_on=False, drop_rep=True, adapt=0.0, seed=0, rounds=None, holdout=False, leaves=15):
    z = np.load(W08 / "rows.npz")
    X, y, meta = z["X"], z["y"], z["meta"]
    keep = np.ones(len(y), bool) if drop_rep else meta[:, 2] == 0
    if patch_on:
        X = np.column_stack([X, z["Xp"]])
    X, y, meta = X[keep], y[keep], meta[keep]
    w = np.ones(len(y), np.float32)
    if adapt > 0:
        R = [r for r in get_pseudo("pseudo2") if r[0] == "validation"]
        Xa = np.concatenate([np.column_stack([r[4], r[5]]) if patch_on else r[4] for r in R])
        ya = np.concatenate([r[6] for r in R]); ka = np.concatenate([r[7] for r in R])
        X = np.concatenate([X, Xa[ka]]); y = np.concatenate([y, ya[ka]]); w = np.concatenate([w, np.full(ka.sum(), adapt, np.float32)])
        print("adapt rows", int(ka.sum()), "positive", float(ya[ka].mean()))
    p = {**PARAMS, "seed": 20261004 + seed, "num_leaves": leaves}
    if holdout:               # Jun-Jan only, early-stopped on February (T08's own protocol); February = test
        feb = meta[:, 1] >= 245
        m0 = lgb.train(p, lgb.Dataset(X[~feb], y[~feb]), 4000, valid_sets=[lgb.Dataset(X[feb], y[feb])],
                       callbacks=[lgb.early_stopping(70, verbose=False)])
        print(name, "holdout best iter", m0.best_iteration, flush=True)
        m0.save_model(str(W08 / f"model_{name}.txt"), num_iteration=m0.best_iteration)
        return m0
    if rounds is None:        # early stopping on February (as T08), then refit on everything at the found size
        feb = (meta[:, 1] >= 245) if adapt == 0 else np.r_[meta[:, 1] >= 243, np.zeros(len(y) - len(meta), bool)]
        tr = ~feb
        m0 = lgb.train(p, lgb.Dataset(X[tr], y[tr], weight=w[tr]), 2400, valid_sets=[lgb.Dataset(X[feb], y[feb])],
                       callbacks=[lgb.early_stopping(70, verbose=False)])
        rounds = int(m0.best_iteration * 1.1)
        print(name, "best iter", m0.best_iteration, "->", rounds, flush=True)
    m = lgb.train(p, lgb.Dataset(X, y, weight=w), rounds)
    m.save_model(str(W08 / f"model_{name}.txt"))
    return m


def predict_pseudo(name: str, d: str) -> pd.DataFrame:
    R = get_pseudo(d)
    b = lgb.Booster(model_file=str(W08 / f"model_{name}.txt"))
    pt = b.num_feature() > 83
    X = np.concatenate([np.column_stack([r[4], r[5]]) if pt else r[4] for r in R])
    p = b.predict(X, num_threads=4)
    out, i = [], 0
    for split, pn, wid, links, X0, Xp, y, kn in R:
        n = len(links)
        out.append(pd.DataFrame({"split": split, "panel": pn, "window_id": wid, "link_id": links, "p": p[i:i + n],
                                 "y": y, "known": kn})); i += n
    return pd.concat(out, ignore_index=True)


def window_iou(D: pd.DataFrame, dec: str = "eiou") -> pd.DataFrame:
    from .models import eiou_topm
    out = []
    for (sp, w), g in D.groupby(["split", "window_id"], sort=False):
        p = g.p.to_numpy()
        if dec == "eiou":
            idx, _ = eiou_topm(p); a = np.zeros(len(p), bool); a[idx] = True
        else:
            a = p >= float(dec)
            if not a.any():
                a[np.argmax(p)] = True
        t = (g.y & g.known).to_numpy(); k = g.known.to_numpy()
        u = ((a & k) | t).sum()
        out.append((sp, g.panel.iloc[0], w, 1.0 if u == 0 else (a & t).sum() / u))
    return pd.DataFrame(out, columns=["split", "panel", "window_id", "iou"])


def compare(D_ref: pd.DataFrame, D_new: pd.DataFrame, dec_ref="eiou", dec_new="eiou"):
    a = window_iou(D_ref, dec_ref).set_index("window_id"); b = window_iou(D_new, dec_new).set_index("window_id")
    rng = np.random.default_rng(0); res = {}
    for sp in ("validation", "private"):
        x = (b.iou - a.iou)[a.split == sp].to_numpy()
        se = np.std([x[rng.integers(0, len(x), len(x))].mean() for _ in range(500)])
        res[sp] = (round(float(b.iou[b.split == sp].mean()), 4), round(float(x.mean()), 4), round(float(se), 4),
                   int((x > 1e-9).sum()), int((x < -1e-9).sum()))
    return res


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "rows":
        build_rows()
    elif cmd == "fit":
        kw = dict(a.split("=") for a in sys.argv[3:])
        fit(sys.argv[2], patch_on=kw.get("patch") == "1", drop_rep=kw.get("drop", "1") == "1",
            adapt=float(kw.get("adapt", 0)), seed=int(kw.get("seed", 0)),
            rounds=int(kw["rounds"]) if "rounds" in kw else None, holdout=kw.get("holdout") == "1",
            leaves=int(kw.get("leaves", 15)))
