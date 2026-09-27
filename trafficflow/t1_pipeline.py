"""Task 1 pooled pipeline: features -> pooled LightGBM -> predictions.

Stages (python -m trafficflow.t1_pipeline <stage> ...):
  feat    per panel: simulate the val/private blackouts on train, sample training
          cells (regular targets + dark cells), extract features for them and for
          every validation/private target cell; write parquet under WORK/feat.
  train   pooled models on all panels: speed, flow (regular), gspeed, gflow (dark
          cells); `--holdout` keeps train days >= HOLD out for scoring.
  predict write WORK/pred/state_<tag>.parquet keyed like the Task 1 templates.
  rounds  print the --rounds JSON of a full fit from holdout tag <tag> (full_rounds).
  ens     state_<tag>.parquet = element-wise mean of state_<m>.parquet, --members m1 m2 ...
TFB_SEED=s (default 0) trains seed-ensemble member s: other CAP row sample and LightGBM seeds.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from .data import PANELS, REL, SLOTS, SPLIT_DAYS
from .t1 import Panel

WORK = Path(os.environ.get("TFB_WORK", "/home/user/work/t1"))
HOLD = 243  # first holdout day of train (last 30 train days)
N_REG = 300_000  # regular training cells per panel
REGIME_NAME = {1: "R1", 2: "R2", 3: "R3"}


def _meta(P: Panel, tt, ll, kind):
    return pd.DataFrame({"panel": P.panel, "t": tt.astype(np.int32), "j": ll.astype(np.int16), "kind": kind,
                         "day": (tt // SLOTS).astype(np.int16),
                         "treg": P.target[tt, ll].astype(np.int8),
                         "y_speed": P.tspeed[tt, ll], "y_flow": P.tflow[tt, ll]})


def feat_panel(panel: str, seed: int = 0):
    out = WORK / "feat"; out.mkdir(parents=True, exist_ok=True)
    if (out / f"{panel}_test.parquet").exists():
        return
    t0 = time.time()
    P = Panel(panel)
    origins = P.select_origins(0, SPLIT_DAYS["train"][1], spacing=36)
    P.apply_blackouts(origins)
    rng = np.random.default_rng(seed)
    ntr = SPLIT_DAYS["train"][1] * SLOTS
    # regular targets (train, not dark)
    tt, ll = np.nonzero((P.target[:ntr] > 0) & ~P.dark[:ntr, None])
    s = rng.choice(len(tt), min(N_REG, len(tt)), replace=False)
    tt, ll = tt[s], ll[s]
    # all train holdout targets are kept for scoring
    ht, hl = np.nonzero((P.target[HOLD * SLOTS:ntr] > 0) & ~P.dark[HOLD * SLOTS:ntr, None])
    ht = ht + HOLD * SLOTS
    s = rng.choice(len(ht), min(150_000, len(ht)), replace=False); ht, hl = ht[s], hl[s]
    keep = tt < HOLD * SLOTS
    tt, ll = np.concatenate([tt[keep], ht]), np.concatenate([ll[keep], hl])
    # dark cells in train (eligible), sampled
    dk = np.nonzero(P.dark[:ntr])[0]
    dt_, dl_ = np.nonzero(P.elig[dk]); dt_ = dk[dt_]
    s = rng.choice(len(dt_), min(400_000, len(dt_)), replace=False); dt_, dl_ = dt_[s], dl_[s]
    kinds = np.array(["reg"] * len(tt) + ["dark"] * len(dt_))
    tt, ll = np.concatenate([tt, dt_]), np.concatenate([ll, dl_])
    parts = []
    for i in range(0, len(tt), 200_000):
        t2, l2 = tt[i:i + 200_000], ll[i:i + 200_000]
        m = _meta(P, t2, l2, "reg"); m["kind"] = kinds[i:i + 200_000]
        f = P.features(t2, l2)
        f64 = f.select_dtypes("float64").columns; f[f64] = f[f64].astype(np.float32)
        parts.append(pd.concat([m, f], axis=1))
    pd.concat(parts, ignore_index=True).to_parquet(out / f"{panel}_train.parquet")
    del parts; gc.collect()
    # test targets: validation + private
    a = SPLIT_DAYS["validation"][0] * SLOTS
    tt, ll = np.nonzero(P.target[a:] > 0); tt = tt + a
    parts = []
    for i in range(0, len(tt), 200_000):
        t2, l2 = tt[i:i + 200_000], ll[i:i + 200_000]
        m = _meta(P, t2, l2, "test")
        m["kind"] = np.where(P.dark[t2], "dark", "reg")
        m["link_id"] = P.links[l2]
        f = P.features(t2, l2)
        f64 = f.select_dtypes("float64").columns; f[f64] = f[f64].astype(np.float32)
        parts.append(pd.concat([m, f], axis=1))
    pd.concat(parts, ignore_index=True).to_parquet(out / f"{panel}_test.parquet")
    print(f"{panel}: feat done in {time.time() - t0:.0f}s, origins={len(origins)}", flush=True)


NON_FEAT = {"panel", "t", "j", "kind", "day", "treg", "y_speed", "y_flow", "link_id", "y_dens"}
USE_FD = os.environ.get("TFB_FD", "0") == "1"
_FD_CACHE: dict = {}


def fd_arrays(panel: str):
    """Per-lane congested-branch parameters (w, k_jam) in milepost order (= column j)."""
    if panel not in _FD_CACHE:
        from .data import network
        from .t1 import mileposts
        net = network(panel)
        o = np.argsort(mileposts(panel, net["links"]))
        fd = net["fd"].iloc[o]
        lanes = fd.lanes.to_numpy(float)
        cap_l = fd.capacity_vph.to_numpy(float) / lanes
        vf = fd.free_speed_kmh.to_numpy(float)
        kj = fd.k_jam.to_numpy(float) / lanes
        w = cap_l / np.maximum(kj - cap_l / np.maximum(vf, 1), 1e-3)
        _FD_CACHE[panel] = (w.astype(np.float32), kj.astype(np.float32))
    return _FD_CACHE[panel]


def add_fd(d: pd.DataFrame) -> pd.DataFrame:
    """Fundamental-diagram features computed from existing feature columns (same maths
    as Panel.features with fd_features=True): congested-branch flow implied by speed,
    speed implied by flow, and q/v density, for the interpolated / previous / next values."""
    w = np.empty(len(d), np.float32); kj = np.empty(len(d), np.float32)
    jj = d["j"].to_numpy()
    for p, idx in d.groupby("panel").indices.items():
        W, KJ = fd_arrays(p)
        w[idx] = W[jj[idx]]; kj[idx] = KJ[jj[idx]]
    vf = d["vf"].to_numpy(np.float32)
    new = {"fd_w": w, "fd_kj": kj}
    for src in ("li", "pv", "nv"):
        v = d[f"{src}_speed"].to_numpy(np.float32); q = d[f"{src}_flow"].to_numpy(np.float32)
        new[f"fd_qc_{src}"] = np.where(v < 0.9 * vf, v * (w * kj / (v + w)), np.nan).astype(np.float32)
        new[f"fd_vc_{src}"] = (q / np.maximum(kj - q / w, 1e-2)).astype(np.float32)
        new[f"fd_k_{src}"] = (q / np.maximum(v, 1)).astype(np.float32)
    for dl in (-1, 1):
        v = d[f"n{dl}_speed_0"].to_numpy(np.float32)
        new[f"fd_qc_n{dl}"] = np.where(v < 0.9 * vf, v * w * kj / (v + w), np.nan).astype(np.float32)
    new["v_over_vf_li"] = (d["li_speed"].to_numpy(np.float32) / vf).astype(np.float32)
    return pd.concat([d, pd.DataFrame(new, index=d.index)], axis=1)


USE_RAMP = os.environ.get("TFB_RAMP", "0") == "1"
_RAMP_CACHE: dict = {}


def ramp_arrays(panel: str):
    """On- and off-ramp flow [T, L] per mainline link in milepost order (column j), valid released ramp
    observations only (NaN where the link has no ramp or no valid value). Ramp flows are released even
    inside the Task 2 blackout rows, where the mainline is blank; Task 1 may use them (offline task)."""
    if panel not in _RAMP_CACHE:
        from .data import load
        from .t1 import mileposts
        d = load(panel)
        links = list(d["links"])
        o = np.argsort(mileposts(panel, links))
        j_of = {links[o[j]]: j for j in range(len(o))}
        meta = d["net"]["ramps"].drop_duplicates("ramp_link_id").set_index("ramp_link_id")
        rf = np.where(d["rvalid"], d["rflow"], np.nan).astype(np.float32)
        T = rf.shape[0]
        on = np.full((T, len(links)), np.nan, np.float32)
        off = np.full((T, len(links)), np.nan, np.float32)
        for ri, r in enumerate(d["ramps"]):
            if r not in meta.index:
                continue
            m = meta.loc[r]
            j = j_of.get(str(m.nearest_mainline_link_id))
            kind = str(m.ramp_type).upper()
            tgt = on if kind in ("OR", "ON") else off if kind in ("FR", "OFF") else None
            if j is None or tgt is None:
                continue
            col = rf[:, ri]
            tgt[:, j] = np.where(np.isnan(tgt[:, j]), col, tgt[:, j] + np.nan_to_num(col))
        _RAMP_CACHE[panel] = (on, off)
    return _RAMP_CACHE[panel]


def _band_sum(a: np.ndarray, t: np.ndarray, j: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """nan-sum of a[t, j+lo .. j+hi] (links outside the corridor ignored); NaN if every value is NaN."""
    L = a.shape[1]
    acc = np.zeros(len(t), np.float32); n = np.zeros(len(t), np.int16)
    for o in range(lo, hi + 1):
        jj = j + o
        ok = (jj >= 0) & (jj < L)
        v = np.full(len(t), np.nan, np.float32)
        v[ok] = a[t[ok], jj[ok]]
        fin = np.isfinite(v)
        acc[fin] += v[fin]; n += fin
    return np.where(n > 0, acc, np.nan).astype(np.float32)


def add_ramp(d: pd.DataFrame) -> pd.DataFrame:
    """Ramp-flow features from the rows' panel, slot t and link j (TFB_RAMP=1): on/off-ramp flow at the
    link, in the 3 links on either side (milepost order, so the tree learns the direction per panel),
    at t-1 / t+1, and for blackout rows the change of the local net injection since the last visible slot."""
    t_all = d["t"].to_numpy(np.int64); j_all = d["j"].to_numpy(np.int64)
    cols = ["rp_on0", "rp_off0", "rp_on_lo3", "rp_on_hi3", "rp_off_lo3", "rp_off_hi3", "rp_on0_prev",
            "rp_on0_next", "rp_off0_prev", "rp_off0_next", "rp_net3", "rp_net3_gap"]
    new = {c: np.full(len(d), np.nan, np.float32) for c in cols}
    dp = d["dark_pos"].to_numpy(np.float64) if "dark_pos" in d else np.zeros(len(d))
    for p, idx in d.groupby("panel").indices.items():
        on, off = ramp_arrays(p)
        T = on.shape[0]
        t = t_all[idx]; j = j_all[idx]
        tp = np.clip(t - 1, 0, T - 1); tn = np.clip(t + 1, 0, T - 1)
        new["rp_on0"][idx] = on[t, j]; new["rp_off0"][idx] = off[t, j]
        new["rp_on0_prev"][idx] = on[tp, j]; new["rp_on0_next"][idx] = on[tn, j]
        new["rp_off0_prev"][idx] = off[tp, j]; new["rp_off0_next"][idx] = off[tn, j]
        new["rp_on_lo3"][idx] = _band_sum(on, t, j, -3, -1); new["rp_on_hi3"][idx] = _band_sum(on, t, j, 1, 3)
        new["rp_off_lo3"][idx] = _band_sum(off, t, j, -3, -1); new["rp_off_hi3"][idx] = _band_sum(off, t, j, 1, 3)
        net = _band_sum(on, t, j, -3, 3) - np.nan_to_num(_band_sum(off, t, j, -3, 3))
        new["rp_net3"][idx] = net
        pos = dp[idx]
        g = np.isfinite(pos) & (pos > 0)
        if g.any():
            tr = np.clip(t[g] - pos[g].astype(np.int64), 0, T - 1)
            ref = _band_sum(on, tr, j[g], -3, 3) - np.nan_to_num(_band_sum(off, tr, j[g], -3, 3))
            v = np.full(len(idx), np.nan, np.float32); v[g] = net[g] - ref
            new["rp_net3_gap"][idx] = v
    return pd.concat([d, pd.DataFrame(new, index=d.index)], axis=1)


CAP = {"reg": (int(os.environ.get("TFB_NREG", 150_000)), 100_000), "dark": (int(os.environ.get("TFB_NDARK", 150_000)), 60_000)}  # (train rows, holdout rows) per panel


def load_train(panels, kind, seed=0):
    rng = np.random.default_rng(seed)
    dfs = []
    for p in panels:
        d = pd.read_parquet(WORK / "feat" / f"{p}_train.parquet")
        d = d[d.kind == kind]
        tr = np.nonzero((d.day < HOLD).to_numpy())[0]; ho = np.nonzero((d.day >= HOLD).to_numpy())[0]
        ctr, cho = CAP[kind]
        tr = rng.choice(tr, min(ctr, len(tr)), replace=False); ho = rng.choice(ho, min(cho, len(ho)), replace=False)
        d = d.iloc[np.sort(np.concatenate([tr, ho]))]
        f64 = d.select_dtypes("float64").columns
        d[f64] = d[f64].astype(np.float32)
        dfs.append(d)
        del d; gc.collect()
    d = pd.concat(dfs, ignore_index=True)
    d["panel_id"] = d.panel.map({p: i for i, p in enumerate(PANELS)}).astype("int16")
    d["y_dens"] = d.y_flow / np.maximum(d.y_speed, 1.0)
    if USE_FD:
        d = add_fd(d)
    if USE_RAMP:
        d = add_ramp(d)
    return d


def base_of(d, c):
    if c == "dens":
        return base_of(d, "flow") / np.maximum(base_of(d, "speed"), 1.0)
    return d[f"li_{c}"].fillna(d[f"h_{c}"]).to_numpy()


# seed-ensemble member (trafficflow/docs/T1_ENSEMBLE.md): changes the CAP row sample in load_train and the
# LightGBM seeds; 0 (default) = the original models bit-for-bit (LightGBM default seeds, rng seed 0)
SEED = int(os.environ.get("TFB_SEED", "0"))


def lgb_seeds(seed: int) -> dict:
    """LightGBM seed parameters of ensemble member `seed`; {} for 0 (LightGBM defaults 1/2/3)."""
    if seed == 0:
        return {}
    return dict(seed=seed, data_random_seed=100 * seed + 1, feature_fraction_seed=100 * seed + 2,
                bagging_seed=100 * seed + 3)


# diversity knobs for ensemble members (defaults = the original models bit-for-bit):
# TFB_LEAVES, TFB_FF (feature_fraction), TFB_MINDATA, TFB_L2, TFB_EXTRA (extra_trees 0/1), TFB_DARK_LEAVES
PARAMS = dict(objective="regression", learning_rate=float(os.environ.get("TFB_LR", 0.1)),
              num_leaves=int(os.environ.get("TFB_LEAVES", 255)),
              min_data_in_leaf=int(os.environ.get("TFB_MINDATA", 100)),
              feature_fraction=float(os.environ.get("TFB_FF", 0.5)), bagging_fraction=0.7, bagging_freq=1,
              lambda_l2=float(os.environ.get("TFB_L2", 2.0)), max_bin=63,
              num_threads=int(os.environ.get("TFB_THREADS", "3")), verbose=-1, **lgb_seeds(SEED))
if os.environ.get("TFB_EXTRA", "0") == "1":
    PARAMS["extra_trees"] = True


def train(panels, holdout: bool, tag: str, rounds: dict | None = None):
    mdir = WORK / "models" / tag; mdir.mkdir(parents=True, exist_ok=True)
    if SEED:  # the hold_*.npy rows follow load_train(..., seed=SEED); t1_holdout.score reads this
        json.dump({"seed": SEED}, open(mdir / "seed.json", "w"))
    json.dump({k: v for k, v in PARAMS.items() if k != "num_threads"}, open(mdir / "params.json", "w"), indent=1)
    report = {}
    print(f"train {tag}: seed {SEED}, lgb seeds {lgb_seeds(SEED) or 'default'}", flush=True)
    for kind, targets in (("reg", ("speed", "flow", "dens")), ("dark", ("speed", "flow", "dens"))):
        d = load_train(panels, kind, seed=SEED)
        feats = [c for c in d.columns if c not in NON_FEAT]
        tr = d.day < HOLD if holdout else np.ones(len(d), bool)
        va = d.day >= HOLD
        for c in targets:
            name = f"{kind}_{c}"
            if (mdir / f"{name}.txt").exists() and (not holdout or (mdir / f"hold_{name}.npy").exists()):
                print("skip", name, flush=True)
                continue
            y = d[f"y_{c}"].to_numpy() - base_of(d, c)
            ok = np.isfinite(y)
            w = (d.length * d.lanes).to_numpy() if c == "dens" else None
            dtr = lgb.Dataset(d.loc[tr & ok, feats], y[tr & ok], weight=None if w is None else w[tr & ok],
                              categorical_feature=["panel_id"], free_raw_data=True)
            p = dict(PARAMS)
            if c == "dens":
                p.update(objective="huber", alpha=float(os.environ.get("TFB_HUBER", 1.0)),
                         num_leaves=int(os.environ.get("TFB_DENS_LEAVES", p["num_leaves"])))
            if kind == "dark":
                p.update(num_leaves=int(os.environ.get("TFB_DARK_LEAVES", 63)), min_data_in_leaf=200,
                         learning_rate=float(os.environ.get("TFB_DARK_LR", 0.05)))
            t0 = time.time()
            if holdout:
                dva = lgb.Dataset(d.loc[va & ok, feats], y[va & ok], weight=None if w is None else w[va & ok], reference=dtr)
                m = lgb.train(p, dtr, int(os.environ.get("TFB_MAXR", 3000)), valid_sets=[dva],
                              callbacks=[lgb.early_stopping(100, verbose=False),
                                                                        lgb.log_evaluation(250)])
                pred = m.predict(d.loc[va, feats], num_iteration=m.best_iteration) + base_of(d[va], c)
                rm = float(np.sqrt(np.nanmean((pred - d.loc[va, f"y_{c}"].to_numpy()) ** 2)))
                report[name] = dict(best_iter=m.best_iteration, rmse=rm)
                np.save(mdir / f"hold_{name}.npy", pred)
            else:
                n = (rounds or {}).get(name, 1500)
                m = lgb.train(p, dtr, n)
            m.save_model(str(mdir / f"{name}.txt"))
            print(name, report.get(name), f"{time.time() - t0:.0f}s", flush=True)
        del d; gc.collect()
    json.dump(report, open(mdir / "report.json", "w"), indent=1)
    return report


def predict(panels, tag: str):
    mdir = WORK / "models" / tag
    out = WORK / "pred"; out.mkdir(parents=True, exist_ok=True)
    models = {n: lgb.Booster(model_file=str(mdir / f"{n}.txt")) for n in
              ("reg_speed", "reg_flow", "reg_dens", "dark_speed", "dark_flow", "dark_dens")}
    frames = []
    for p in panels:
        d = pd.read_parquet(WORK / "feat" / f"{p}_test.parquet")
        d["panel_id"] = np.int16(PANELS.index(p))
        if USE_FD:
            d = add_fd(d)
        if USE_RAMP:
            d = add_ramp(d)
        feats = models["reg_speed"].feature_name()
        sp = np.empty(len(d)); fl = np.empty(len(d)); dn = np.empty(len(d))
        for kind in ("reg", "dark"):
            m = (d.kind == kind).to_numpy()
            if m.any():
                X = d.loc[m, feats]
                nt = int(os.environ.get("TFB_PRED_THREADS", "1"))
                sp[m] = models[f"{kind}_speed"].predict(X, num_threads=nt) + base_of(d[m], "speed")
                fl[m] = models[f"{kind}_flow"].predict(X, num_threads=nt) + base_of(d[m], "flow")
                dn[m] = models[f"{kind}_dens"].predict(X, num_threads=nt) + base_of(d[m], "dens")
        frames.append(pd.DataFrame({"panel": p, "t": d.t, "link_id": d.link_id, "regime": d.treg,
                                    "kind": d.kind, "speed": sp, "flow_lane": fl, "dens_lane": dn,
                                    "lanes": d.lanes}))
    res = pd.concat(frames, ignore_index=True)
    res.to_parquet(out / f"state_{tag}.parquet")
    return res


def full_rounds(report: dict, maxr: int = 3000) -> dict:
    """Rounds of a full-data fit from a holdout report (the rule behind full3): 1.1 x the best iteration,
    rounded to 10; a model whose early stopping ran into the round cap (best >= maxr - 20) gets 1.1 x maxr."""
    return {k: int(round(1.1 * (maxr if v["best_iter"] >= maxr - 20 else v["best_iter"]), -1))
            for k, v in report.items()}


def ensemble(tag: str, members) -> None:
    """WORK/pred/state_<tag>.parquet = element-wise mean of state_<m>.parquet over `members` for speed,
    flow_lane and dens_lane; every other column must be identical (same rows in the same order)."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    out = WORK / "pred"
    vals = ("speed", "flow_lane", "dens_lane")
    base = pq.read_table(out / f"state_{members[0]}.parquet")
    acc = {c: base.column(c).to_numpy().astype(np.float64) for c in vals}
    nan = {c: int(np.isnan(acc[c]).sum()) for c in vals}
    for m in members[1:]:
        t = pq.read_table(out / f"state_{m}.parquet")
        assert t.column_names == base.column_names, f"{m}: columns {t.column_names} vs {base.column_names}"
        for c in base.column_names:
            if c not in vals:
                assert t.column(c).equals(base.column(c)), f"{m}: column {c} differs from {members[0]}"
        for c in vals:
            x = t.column(c).to_numpy()
            nan[c] += int(np.isnan(x).sum())
            acc[c] += x
        del t
    for c in vals:
        base = base.set_column(base.column_names.index(c), c, pa.array(acc[c] / len(members)))
    pq.write_table(base, out / f"state_{tag}.parquet")
    print(f"state_{tag}: mean of {list(members)}, {base.num_rows} rows, NaN in members {nan}", flush=True)


def ensemble_weighted(tag: str, new: str, others, w_reg: float, w_dark: float) -> None:
    """WORK/pred/state_<tag>.parquet: member `new` weighted w_reg on regular rows and w_dark on blackout
    ("dark") rows; the `others` share the remaining weight equally (trafficflow/t1_weighted_eval.py)."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    out = WORK / "pred"
    vals = ("speed", "flow_lane", "dens_lane")
    base = pq.read_table(out / f"state_{new}.parquet")
    kind = base.column("kind").to_numpy(zero_copy_only=False)
    w = np.where(kind == "dark", w_dark, w_reg)
    rest = {c: np.zeros(base.num_rows) for c in vals}
    for m in others:
        t = pq.read_table(out / f"state_{m}.parquet")
        for c in base.column_names:
            if c not in vals:
                assert t.column(c).equals(base.column(c)), f"{m}: column {c} differs from {new}"
        for c in vals:
            rest[c] += t.column(c).to_numpy() / len(others)
        del t
    for c in vals:
        v = w * base.column(c).to_numpy() + (1 - w) * rest[c]
        base = base.set_column(base.column_names.index(c), c, pa.array(v))
    pq.write_table(base, out / f"state_{tag}.parquet")
    print(f"state_{tag}: {new} x (reg {w_reg}, dark {w_dark}) + {list(others)}; dark rows {int((kind == 'dark').sum())}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage")
    ap.add_argument("--panels", nargs="*", default=PANELS)
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--holdout", action="store_true")
    ap.add_argument("--rounds", default="{}", help='JSON {"reg_speed": 950, ...} for full fits')
    ap.add_argument("--members", nargs="*", help="ens: state tags to average into state_<tag>")
    ap.add_argument("--new", help="ensw: member weighted per kind")
    ap.add_argument("--w-reg", type=float, default=0.25)
    ap.add_argument("--w-dark", type=float, default=0.25)
    a = ap.parse_args()
    if a.stage == "feat":
        for p in a.panels:
            feat_panel(p)
            gc.collect()
    elif a.stage == "train":
        print(train(a.panels, a.holdout, a.tag, json.loads(a.rounds)))
    elif a.stage == "predict":
        predict(a.panels, a.tag)
    elif a.stage == "rounds":  # full-fit --rounds JSON from a holdout tag's report
        print(json.dumps(full_rounds(json.load(open(WORK / "models" / a.tag / "report.json")))))
    elif a.stage == "ens":
        ensemble(a.tag, a.members)
    elif a.stage == "ensw":
        ensemble_weighted(a.tag, a.new, a.members, a.w_reg, a.w_dark)
