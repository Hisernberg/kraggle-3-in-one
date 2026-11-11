"""T08 onset (anubisyy public notebook, v20 assets: onset_lgb.txt + onset.cbm, 0.5/0.5, thr 0.55, top-1 fallback)
reproduced so it can be scored on the test-month pseudo onset windows, plus our own retrained variants.

The T08 features (``queue_features.Panel.features``) only read the 12-slot history [T-12, T) of speed, flow and
eligibility, so they are computed identically for official windows (window_history.parquet) and for pseudo windows
(the filled masked-view history stored in pseudo3/windows_<panel>.npz).

    python -m trafficflow.t2.onset_t08 official OUT.parquet     # p per official onset window x link (T+30)
    python -m trafficflow.t2.onset_t08 pseudo OUT.parquet       # p per pseudo3 onset window x link, with labels

Env: T08_ASSETS (onset_lgb.txt, onset.cbm: released model files, loaded as data), TFB_DATA_ROOT.\nFeatures: our re-implementation in t08_features.py.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ..data import SLOTS, T0
from .core import PANELS8, statics

REL = Path(os.environ.get("TFB_DATA_ROOT", "/home/urad/tfbroot/data/kaggle_public"))
from .t08_features import HISTORY, T08Panel as Panel, panel_names

ASSETS = Path(os.environ.get("T08_ASSETS", "/home/urad/tfbroot/work/t08/assets"))
PSEUDO = Path(os.environ.get("T08_PSEUDO", "/home/urad/tfbroot/work/t2/pseudo3"))


def panels():
    names = panel_names()
    return {n: Panel(n, i) for i, n in enumerate(names)}


def feats(P: Panel, speed, flow, elig, origin) -> np.ndarray:
    """[L, 84] T08 feature rows at horizon step 6 (the only scored onset step)."""
    return P.onset_rows(speed, flow, elig, origin)


def official(out: Path):
    PS = panels(); rows = []
    for name, P in PS.items():
        for split in ("validation", "private"):
            folder = REL / "task2" / name / split
            W = pd.read_csv(folder / "window_index.csv")
            Hh = pd.read_parquet(folder / "window_history.parquet")
            Hh["link_id"] = Hh.link_id.astype(str)
            for w in W[W.condition == "queue_onset"].itertuples(index=False):
                origin = pd.Timestamp(w.forecast_origin)
                times = pd.date_range(end=origin - pd.Timedelta(minutes=5), periods=HISTORY, freq="5min",
                                      tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
                h = Hh[Hh.window_id == w.window_id]
                m = lambda c: h.pivot(index="timestamp", columns="link_id", values=c).reindex(
                    index=times, columns=P.links).to_numpy(dtype=np.float32)
                X = feats(P, m("speed_kmh"), m("flow_vph"), m("is_score_eligible"), origin)
                rows.append((split, name, w.window_id, origin, P.links, X, None, None))
    return rows


def pseudo(out: Path):
    PS = panels(); rows = []
    for name, P in PS.items():
        st = statics(name)
        perm = np.array([st["lid"][l] for l in P.links])           # T08 link order -> our cache order
        M = pd.read_parquet(PSEUDO / f"meta_{name}.parquet").set_index("w")
        z = np.load(PSEUDO / f"windows_{name}.npz")
        for w in M.index[M.condition == "queue_onset"]:
            T = int(M.loc[w, "T"]); origin = T0 + pd.Timedelta(minutes=5 * T)
            hs, hf, he = z["hs"][w][:, perm], z["hf"][w][:, perm], z["he"][w][:, perm].astype(np.float32)
            X = feats(P, hs, hf, he, origin)
            y = z["y"][w][:, perm]; kn = z["known"][w][:, perm]
            rows.append((M.loc[w, "split"], name, M.loc[w, "window_id"], origin, P.links, X, y, kn))
    return rows


def predict(rows, models):
    """models: list of (callable X -> p, weight). Returns long frame: split, panel, window_id, link_id, li, p."""
    out = []
    Xall = np.concatenate([r[5] for r in rows])
    p = sum(w * f(Xall) for f, w in models) / sum(w for _, w in models)
    i = 0
    for split, name, wid, origin, links, X, y, kn in rows:
        n = len(links)
        d = pd.DataFrame({"split": split, "panel": name, "window_id": wid, "link_id": links, "li": np.arange(n),
                          "p": p[i:i + n]})
        if y is not None:
            d["y_any"] = y[:-1].any(0)            # queued at T+5..T+25 (should be ~never for onset windows)
            d["y"] = y[-1]; d["known"] = kn[-1]; d["known_early"] = kn[:-1].any(0)
        out.append(d); i += n
    return pd.concat(out, ignore_index=True)


def t08_models():
    import lightgbm as lgb
    from catboost import CatBoostClassifier
    b = lgb.Booster(model_file=str(ASSETS / "onset_lgb.txt"))
    c = CatBoostClassifier(); c.load_model(str(ASSETS / "onset.cbm"))
    return [(lambda X: b.predict(X, num_threads=4), 0.5), (lambda X: c.predict_proba(X, thread_count=4)[:, 1], 0.5)]


def decode_t08(g: pd.DataFrame, thr=0.55) -> np.ndarray:
    s = g.p.to_numpy(); sel = s >= thr
    if not sel.any():
        sel[np.argmax(s)] = True
    return sel


def iou_table(D: pd.DataFrame, pred_col: str) -> pd.DataFrame:
    """Window IoU at T+30 over known cells (pseudo labels)."""
    g = D.assign(a=D[pred_col].astype(bool), t=D.y.astype(bool) & D.known.astype(bool), k=D.known.astype(bool))
    r = g.groupby(["split", "panel", "window_id"]).apply(
        lambda h: pd.Series({"inter": int((h.a & h.t).sum()), "union": int(((h.a & h.k) | h.t).sum())}))
    r["iou"] = np.where(r.union > 0, r.inter / r.union.clip(lower=1), 1.0)
    return r.reset_index()


if __name__ == "__main__":
    mode, out = sys.argv[1], Path(sys.argv[2])
    rows = official(out) if mode == "official" else pseudo(out)
    D = predict(rows, t08_models())
    D["t08"] = False
    for _, g in D.groupby("window_id"):
        D.loc[g.index, "t08"] = decode_t08(g)
    D.to_parquet(out)
    print(mode, D.window_id.nunique(), "windows,", int(D.t08.sum()), "T08 cells")
