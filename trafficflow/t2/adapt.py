"""Task 2 online adaptation: ongoing components trained on train windows plus test-month pseudo windows.

For a window at origin T the organizer's rule (forum #742068) allows any released data with timestamp <= T. Every
April window therefore may use all of March's masked view, so the March pseudo windows of ``pseudo.py`` are legal
training data for the April forecasts (and the pseudo-holdout's April windows measure the effect honestly: disjoint
month, no April data in training). Only the rows whose label is known (observed eligible horizon cells) are used.

    T2_WORK=/home/user/work/t2 T2_FEAT=/home/user/work/t2/feat_v3 \\
        python -m trafficflow.t2.adapt train og_v3 p2 validation 1.0     # -> WORK/adapt/model_ad_og_v3_p2_validation_1.0_queue_ongoing.txt
"""
from __future__ import annotations

import gc
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from .core import K, PANELS8, WORK
from .cv import gather, window_weights
from .models import CFG
from .pseudo import OUT as PSEUDO
from .robust import CAND_FRAC, VARIANTS

ADAPT = WORK / "adapt"
COND = "queue_ongoing"


def pseudo_rows(cols: list[str], split: str):
    """Rows of the pseudo windows of ``split`` (one condition) with a known label, in the column order ``cols``."""
    Xs, ys, gws = [], [], []
    for p in PANELS8:
        pc = PANELS8.index(p)
        M = pd.read_parquet(PSEUDO / f"meta_{p}.parquet").set_index("w")
        ws = M.index[(M.condition == COND) & (M.split == split)].to_numpy()
        F = pd.read_parquet(PSEUDO / f"feat_{p}.parquet", columns=[c for c in cols if c != "pcode"] + ["w", "k", "link"],
                            filters=[("w", "in", ws.tolist())])
        z = np.load(PSEUDO / f"windows_{p}.npz")
        w = F.w.to_numpy().astype(np.int64); k = F.k.to_numpy().astype(np.int64) - 1; li = F.link.to_numpy().astype(np.int64)
        known = z["known"][w, k, li]
        y = z["y"][w, k, li]
        F = F[known]
        F["pcode"] = np.float32(pc)
        Xs.append(F[cols].to_numpy(np.float32)); ys.append(y[known].astype(np.int8))
        gws.append(10_000_000 + pc * 100000 + w[known])
        del F
        gc.collect()
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(gws)


def train(variant: str, cfg: str, split: str, alpha: float, seed: int = 0) -> str:
    t0 = time.time()
    params, rounds = CFG[cfg]
    R, _ = gather(COND, cand_frac=CAND_FRAC, drop=VARIANTS[variant])
    Xp, yp, gwp = pseudo_rows(list(R.cols), split)
    n0 = len(R.y)
    print(f"train rows {n0}, pseudo rows {len(yp)} (positive {yp.mean():.3f}), alpha {alpha}", flush=True)
    wt = window_weights(np.concatenate([R.gw, gwp]))
    wt[n0:] *= alpha
    X = np.concatenate([R.X, Xp]); y = np.concatenate([R.y, yp]).astype(np.float32)
    cols = list(R.cols)
    del R, Xp
    gc.collect()
    ds = lgb.Dataset(X, y, feature_name=cols, weight=wt, params={"verbose": -1, "max_bin": params.get("max_bin", 255)}).construct()
    del X
    gc.collect()
    m = lgb.train({**params, "seed": seed}, ds, rounds)
    ADAPT.mkdir(parents=True, exist_ok=True)
    name = f"ad_{variant}_{cfg}_{split}_{alpha}" + (f"_s{seed}" if seed else "")
    m.save_model(str(ADAPT / f"model_{name}_{COND}.txt"))
    print(f"saved {name} in {time.time() - t0:.0f}s", flush=True)
    return name


if __name__ == "__main__":
    if sys.argv[1] == "train":
        train(sys.argv[2], sys.argv[3], sys.argv[4], float(sys.argv[5]), int(sys.argv[6]) if len(sys.argv) > 6 else 0)
