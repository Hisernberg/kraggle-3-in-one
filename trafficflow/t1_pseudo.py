"""Test-month pseudo-holdout for Task 1: hide observed cells of March (validation) and April (private) and predict them.

The train holdout (last 30 train days) shares the train simulation's demand draw; the test months are independent
draws. Capacity/data gains measured on the train holdout transferred at 15-50% (H7, H8), variance reduction at 85-100%
(EXPERIMENTS.md, 28 Sep). Observed, eligible, non-target cells of the test months are released data; hiding a small
random sample of them (about 1% of the observed cells) and predicting it the way the pipeline predicts a regular
target cell measures generalization to the test months' draws directly. No hidden label is used.

    python -m trafficflow.t1_pseudo build [panels]            features + truth of the pseudo cells -> WORK/pseudo/<panel>.parquet
    python -m trafficflow.t1_pseudo predict full3,full9,...    regular-model predictions -> WORK/pseudo/pred_<tag>_<panel>.npz
    python -m trafficflow.t1_pseudo score                      S_state of the adopted schemes per month

Pseudo cells are regular (non-blackout) cells; blackout rows have no observed cells.
"""
from __future__ import annotations

import gc
import os
import sys
import time

import numpy as np
import pandas as pd

from .data import PANELS, SLOTS, SPLIT_DAYS
from .t1 import Panel
from .t1_pipeline import WORK, add_fd, add_ramp, base_of


def train_blackout_panel(panel: str) -> Panel:
    """The Panel exactly as feat_panel builds it: train-split Task 2 origins blacked out (rows T+1..T+18)."""
    P = Panel(panel)
    P.apply_blackouts(P.select_origins(0, SPLIT_DAYS["train"][1], spacing=36))
    return P

OUT = WORK / "pseudo"
N_PER_MONTH = int(os.environ.get("TFB_PSEUDO_N", 20_000))
SPEED_W, FLOW_W = 0.54, 0.46


def build_panel(panel: str, seed: int = 0) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / f"{panel}.parquet"
    if f.exists():
        return
    t0 = time.time()
    P = train_blackout_panel(panel)
    rng = np.random.default_rng(seed + 17 * PANELS.index(panel))
    picks = []
    for split in ("validation", "private"):
        a, b = SPLIT_DAYS[split]
        sl = slice(a * SLOTS, b * SLOTS)
        ok = (np.isfinite(P.X["speed"][sl]) & np.isfinite(P.X["flow"][sl]) & P.elig[sl] & (P.target[sl] == 0)
              & ~P.dark[sl][:, None])
        tt, ll = np.nonzero(ok)
        s = rng.choice(len(tt), min(N_PER_MONTH, len(tt)), replace=False)
        picks.append((split, tt[s] + a * SLOTS, ll[s]))
    tt = np.concatenate([p[1] for p in picks]); ll = np.concatenate([p[2] for p in picks])
    split = np.concatenate([[p[0]] * len(p[1]) for p in picks])
    y_speed = P.X["speed"][tt, ll].astype(np.float32).copy()
    y_flow = P.X["flow"][tt, ll].astype(np.float32).copy()          # per lane (Panel stores flow / lanes)
    for c in P.X:                                                    # hide the pseudo cells, as a target is hidden
        P.X[c][tt, ll] = np.nan
    P._interp()
    X = P.features(tt, ll)
    X["panel_id"] = np.int16(PANELS.index(panel)); X["panel"] = panel; X["j"] = ll; X["t"] = tt
    X = add_fd(X)
    X = add_ramp(X)          # every member's features (full7 uses the ramp columns; the others ignore them)
    X["split"] = split
    X["regime_day"] = P.regime_day[tt // SLOTS].astype(np.int8)
    X["y_speed"] = y_speed; X["y_flow"] = y_flow
    X["lanes_"] = P.lanes[ll]; X["vf_"] = P.vf[ll]
    f64 = X.select_dtypes("float64").columns
    X[f64] = X[f64].astype(np.float32)
    X.to_parquet(f)
    print(f"{panel}: {len(X)} pseudo cells, {time.time() - t0:.0f}s", flush=True)


def build_train_rows(panel: str, rounds: int = 3, n_per_round: int = 30_000, seed: int = 1) -> None:
    """Transductive training rows: observed, eligible, non-target, non-blackout cells of the test months, disjoint
    from the evaluation pseudo cells. Each round hides its own cells (about 5% of the observed cells, so the neighbour
    pattern stays close to a real target's), builds their features, and restores the panel. Columns match the feat
    tables (load_train adds FD / ramp columns), kind 'reg', flagged `pseudo`."""
    f = OUT / f"{panel}_trainrows.parquet"
    if f.exists():
        return
    t0 = time.time()
    P = train_blackout_panel(panel)
    ev = pd.read_parquet(OUT / f"{panel}.parquet", columns=["t", "j"])
    ev_key = set(zip(ev.t.to_numpy().tolist(), ev.j.to_numpy().tolist()))
    orig = {c: P.X[c].copy() for c in P.X}
    rng = np.random.default_rng(seed + 31 * PANELS.index(panel))
    parts = []
    for split in ("validation", "private"):
        a, b = SPLIT_DAYS[split]
        sl = slice(a * SLOTS, b * SLOTS)
        ok = (np.isfinite(orig["speed"][sl]) & np.isfinite(orig["flow"][sl]) & P.elig[sl] & (P.target[sl] == 0)
              & ~P.dark[sl][:, None])
        tt, ll = np.nonzero(ok); tt = tt + a * SLOTS
        keep = np.array([(int(t), int(l)) not in ev_key for t, l in zip(tt, ll)])
        tt, ll = tt[keep], ll[keep]
        order = rng.permutation(len(tt))
        for r in range(rounds):
            s = order[r * n_per_round:(r + 1) * n_per_round]
            t_r, l_r = tt[s], ll[s]
            for c in P.X:
                P.X[c] = orig[c].copy()
                P.X[c][t_r, l_r] = np.nan
            P._interp()
            F = P.features(t_r, l_r)
            m = pd.DataFrame({"panel": panel, "t": t_r.astype(np.int32), "j": l_r.astype(np.int16), "kind": "reg",
                              "day": (t_r // SLOTS).astype(np.int16), "treg": P.regime_day[t_r // SLOTS].astype(np.int8),
                              "y_speed": orig["speed"][t_r, l_r], "y_flow": orig["flow"][t_r, l_r]})
            f64 = F.select_dtypes("float64").columns
            F[f64] = F[f64].astype(np.float32)
            parts.append(pd.concat([m, F], axis=1))
    for c in P.X:
        P.X[c] = orig[c]
    X = pd.concat(parts, ignore_index=True)
    X.to_parquet(f)
    print(f"{panel}: {len(X)} transductive rows, {time.time() - t0:.0f}s", flush=True)


def predict(tags, panels=PANELS, threads: int = 4) -> None:
    import lightgbm as lgb
    for tag in tags:
        models = {c: lgb.Booster(model_file=str(WORK / "models" / tag / f"reg_{c}.txt")) for c in ("speed", "flow", "dens")}
        for panel in panels:
            f = OUT / f"pred_{tag}_{panel}.npz"
            if f.exists():
                continue
            X = pd.read_parquet(OUT / f"{panel}.parquet")
            pr = {c: m.predict(X[m.feature_name()], num_threads=threads) + base_of(X, c) for c, m in models.items()}
            np.savez_compressed(f, **pr)
            del X; gc.collect()
        print(f"{tag}: predicted", flush=True)


def load(panel: str, tags) -> tuple[pd.DataFrame, dict]:
    X = pd.read_parquet(OUT / f"{panel}.parquet", columns=["split", "regime_day", "y_speed", "y_flow", "lanes_", "vf_"])
    P = {t: dict(np.load(OUT / f"pred_{t}_{panel}.npz")) for t in tags}
    return X, P


def s_state(v, q, ys, yq, reg) -> float:
    """Official Task 1 score (per regime, then mean); flow already per lane."""
    out = []
    for r in (1, 2, 3):
        m = reg == r
        if not m.any():
            continue
        rs = np.sqrt(np.mean((v[m] - ys[m]) ** 2)); rq = np.sqrt(np.mean((q[m] - yq[m]) ** 2))
        out.append(SPEED_W * max(0, 1 - rs / 25) + FLOW_W * max(0, 1 - rq / 600))
    return float(np.mean(out))


SCHEMES = {
    "H5w": [("full3", 0.25), ("full4", 0.25), ("full5", 0.25), ("full7", 0.25)],
    "H7": [("full9", 0.5), ("full3", 0.125), ("full4", 0.125), ("full5", 0.125), ("full7", 0.125)],
    "H8": [("full9", 0.5), ("full10", 0.5)],
}


def score(schemes=SCHEMES, panels=PANELS, gate: float = 0.6, a: float = 0.75) -> pd.DataFrame:
    """S_state per panel and month for each scheme, with the adopted gated reconciliation (TV smoothing is left out:
    pseudo cells are isolated, where the smoothing only moves run ends)."""
    from .t1_holdout import reconcile
    tags = sorted({t for v in schemes.values() for t, _ in v})
    rows = []
    for panel in panels:
        X, P = load(panel, tags)
        for name, members in schemes.items():
            pr = {c: sum(w * P[t][c] for t, w in members) for c in ("speed", "flow", "dens")}
            v, q = reconcile(pr["speed"], pr["flow"], pr["dens"], a)
            g = pr["speed"] < gate * X.vf_.to_numpy()
            v = np.where(g, v, pr["speed"]); q = np.where(g, q, pr["flow"])
            v = np.clip(v, 3.0, 130.0); q = np.maximum(q * X.lanes_.to_numpy(), 60.0) / X.lanes_.to_numpy()
            for split in ("validation", "private"):
                m = (X.split == split).to_numpy()
                rows.append(dict(panel=panel, split=split, scheme=name,
                                 S_state=s_state(v[m], q[m], X.y_speed.to_numpy()[m], X.y_flow.to_numpy()[m],
                                                 X.regime_day.to_numpy()[m]),
                                 rmse_v=float(np.sqrt(np.mean((v[m] - X.y_speed.to_numpy()[m]) ** 2))),
                                 rmse_q=float(np.sqrt(np.mean((q[m] - X.y_flow.to_numpy()[m]) ** 2)))))
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "score.csv", index=False)
    for split in ("validation", "private"):
        piv = d[d.split == split].pivot_table(index="scheme", columns="panel", values="S_state").reindex(list(schemes))
        base = piv.iloc[0]
        print(split, "S_state (0.35 x delta = S_total units):")
        print(piv.assign(mean=piv.mean(1), dS=(piv - base).mean(1), dTotal=0.35 * (piv - base).mean(1),
                         up=(piv > base).sum(1)).round(5).to_string())
    return d


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "build":
        for p in (sys.argv[2:] or PANELS):
            build_panel(p); gc.collect()
    elif cmd == "trainrows":
        for p in (sys.argv[2:] or PANELS):
            build_train_rows(p); gc.collect()
    elif cmd == "predict":
        predict(sys.argv[2].split(","), sys.argv[3:] or PANELS)
    elif cmd == "score":  # score ['{"name": [["tag", w], ...], ...}']  (default: SCHEMES; the first is the reference)
        import json
        spec = json.loads(sys.argv[2]) if len(sys.argv) > 2 else None
        score({k: [(t, float(w)) for t, w in v] for k, v in spec.items()} if spec else SCHEMES)
