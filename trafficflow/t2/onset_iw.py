"""Covariate-shift-adapted onset models (TASK2_ANALYSIS.md section 18).

The section 17 weight model (``shift_cv``) is applied to every onset training window (cand, sim, off;
the t2h re-drawn set), one weight vector per target month:
* the fitted classifier is the cross-validated L2 logistic regression of the target month's 40 onset
  windows against the 2,081 CV sim windows (panel fixed effects, data <= T only, ``shift_cv._cached``,
  full-data coefficients);
* the features are the same ``shift_cv`` window features, standardised with the sim windows' mean / SD;
* ``w = exp(alpha * x.beta)`` (tempered, alpha = 0.5 by default), normalised to mean 1 within each
  panel over all training windows, capped at CLIP x the panel mean and renormalised
  (``shift_cv.clip_normalise``).

Every row of a window (T+30 x every link) carries the window's weight, so each panel keeps its
unweighted share of the training loss. The v8 onset recipe (on_v3 seeds 0-5 and on_v2 seeds 0-2 at
0.75 / 0.25, p1, location prior, hybrid labels, 4 week folds; stage 2 = ``stack.panel_features`` on the
stage-1 profile, 3 seeds, blended at 0.3) is trained with these row weights in both stages. The
validation-weighted models predict the validation rows, the private-weighted models the private rows.

    export PYTHONPATH=/home/user/knee OMP_NUM_THREADS=2 T2_THREADS=2 T2_WORK=/home/user/work/t2h T2_FEAT=/home/user/work/t2h/feat
    python -m trafficflow.t2.onset_iw weights                         # window features, weights, ESS report
    python -m trafficflow.t2.onset_iw cv VARIANT SEED TARGET [--alpha A]   # weighted 4-fold OOF, all onset rows
    python -m trafficflow.t2.onset_iw cheap                           # single-seed test vs unweighted seeds
    python -m trafficflow.t2.onset_iw stack TARGET [--alpha A]        # weighted stage 2 on the weighted seeds9 mean
    python -m trafficflow.t2.onset_iw table [--alpha A]               # full recipe vs v8, gates 1-2
    python -m trafficflow.t2.onset_iw build NAME [--alpha A]          # full-train models, footprint, gate, candidate

Outputs go to T2_IW_OUT (default /home/user/work/t2iw). A candidate that passes every gate is written to
/home/user/work/t2/lgb_v11_<NAME>.csv (lgb_v8_seeds9_stack03.csv with only the onset rows replaced) and
/home/user/work/t2h/probs_v11_<NAME>_onset.parquet (layout and row order of the v8 onset probabilities).
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import resource
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .core import K, PANELS8, WORK, official_windows, statics
from . import shift_cv as sc
from .models import eiou_topm

OUT = Path(os.environ.get("T2_IW_OUT", "/home/user/work/t2iw"))
T2H = Path("/home/user/work/t2h")
CSV_DIR = Path("/home/user/work/t2")
BASE_CSV = CSV_DIR / "lgb_v8_seeds9_stack03.csv"
V8_PROBS = T2H / "probs_v8_seeds9_stack03_onset.parquet"
COND = "queue_onset"
TARGETS = ("validation", "private")
SPECS = [("on_v3", s, 3.0) for s in range(6)] + [("on_v2", s, 2.0) for s in range(3)]   # v8: 0.75 / 0.25
STACK_W = 0.3
STACK_SEEDS = (0, 1, 2)
KEY = ["window_id", "panel", "k", "link"]
FEAT_COLS = ["rmin", "nslow", "tod", "dow", "visT", "orec", "orq7", "fmax", "fext", "qobs", "eq", "nsite", "sur",
             "m", "pmax", "psum", "cov"]
BAL_COLS = ("sur", "orq7", "lpsum", "orec", "cov", "wkend")


def atag(alpha: float) -> str:
    return f"a{int(round(alpha * 10)):02d}"          # 0.5 -> a05, 1.0 -> a10


def _env_ok():
    assert Path(WORK) == T2H, "run with T2_WORK=/home/user/work/t2h T2_FEAT=/home/user/work/t2h/feat"
    assert not os.environ.get("T2_RELABEL") and not os.environ.get("T2_TRUTH"), "unset T2_RELABEL / T2_TRUTH"


# ----------------------------------------------------------------------------
# window features and importance weights of the training windows
def train_window_features() -> pd.DataFrame:
    """``shift_cv`` onset window features (same functions and inputs: t2h feature tables at T+30, v6 OOF
    stage-1 probabilities, history coverage) for every onset training window: cand, sim and off. The sim /
    off rows are checked against shift_cv's cached table. Cached as OUT/winfeat_onset_train.parquet."""
    f = OUT / "winfeat_onset_train.parquet"
    if f.exists():
        return pd.read_parquet(f)
    S1 = sc._cv_stage1(COND)
    rows = []
    for p in PANELS8:
        pc = PANELS8.index(p)
        L = statics(p)["L"]
        z = np.load(sc.WORK_ON / f"ds_{p}.npz", allow_pickle=True)
        cov = z["w_cov"]
        X, M = sc._tables(sc.FEAT_ON, p, None, sc.ON_COLS, float(K))
        M = M[M.condition == COND].drop_duplicates("w").set_index("w")
        X = X[X.w.isin(set(M.index))]
        s1 = S1[S1.gw // 100000 == pc]
        s1g = {g: d for g, d in s1.groupby("gw")}
        for w, g in X.groupby("w", sort=False):
            w = int(w)
            key = pc * 100000 + w
            q = s1g[key].sort_values(["k", "link"])
            g = g.sort_values("link")
            pv = np.zeros(L); pv[q.link.to_numpy().astype(int)] = q.p.to_numpy()
            r = sc._on_window(g, pv[g.link.to_numpy().astype(int)])
            r.update(sc._stage1_window(q.p.to_numpy()))
            r["cov"] = float(cov[w])
            r.update(key=key, panel=p, src=M.src[w], fold=int(M.fold[w]))
            rows.append(r)
        del z
    D = pd.DataFrame(rows)
    C = sc.window_features(COND)
    C = C[C.src.isin(["sim", "off"])]
    T = D.set_index("key").loc[C.key.astype(np.int64).to_numpy()]
    for c in FEAT_COLS:
        assert np.allclose(T[c].to_numpy(np.float64), C[c].to_numpy(np.float64), rtol=1e-9, atol=1e-12,
                           equal_nan=True), f"feature {c} differs from shift_cv's table"
    assert (T.src.to_numpy() == C.src.to_numpy()).all() and (T.panel.to_numpy() == C.panel.to_numpy()).all()
    OUT.mkdir(parents=True, exist_ok=True)
    D.to_parquet(f)
    return D


@lru_cache(None)
def _weights(target: str, alpha: float, clip: float) -> tuple[pd.Series, dict]:
    r = sc._cached(COND, target)                       # the section 17 fit (lambda by CV, full-data coef)
    D0, X0 = r["D"], r["X"]
    sim0 = (D0.src == "sim").to_numpy()
    coef = r["coef"]
    mu, sd = X0[sim0].mean(), X0[sim0].std().replace(0, 1.0)
    T = train_window_features()
    XT = sc.design(T, COND)                             # NaN -> median of the (same) sim windows
    Z = ((XT[coef.index] - mu[coef.index]) / sd[coef.index]).to_numpy(np.float64)
    eta = Z @ coef.to_numpy(np.float64)
    w, share = sc.clip_normalise(np.exp(alpha * (eta - eta.max())), T.panel.to_numpy(), clip)
    W = pd.Series(w, index=T.key.astype(np.int64).to_numpy(), name="w")
    return W, dict(eta=eta, clipped=share, T=T, XT=XT, fit=r)


def train_weights(target: str, alpha: float = 0.5, clip: float = sc.CLIP) -> pd.Series:
    """Tempered importance weights toward ``target`` of every onset training window (pd.Series by gw, mean 1
    within panel)."""
    return _weights(target, float(alpha), float(clip))[0]


def _kish(w: np.ndarray) -> float:
    return float(w.sum() ** 2 / (w ** 2).sum()) if len(w) else float("nan")


def weight_report(alphas=(0.5, 1.0)) -> pd.DataFrame:
    """ESS / clipping of the training weights, the agreement of the full-data fit with shift_cv's cross-fitted
    log-odds on the sim windows, and the balance of the training mix toward the target (within-panel SMD,
    target - training, in SD of the sim windows; unweighted > weighted)."""
    rows = []
    for tg in TARGETS:
        for a in alphas:
            W, d = _weights(tg, float(a), sc.CLIP)
            T = d["T"]; pan = T.panel.to_numpy(); src = T.src.to_numpy(); w = W.to_numpy()
            per = {p: _kish(w[pan == p]) for p in PANELS8}
            npn = {p: int((pan == p).sum()) for p in PANELS8}
            pmin = min(per, key=lambda p: per[p] / npn[p])
            fit = d["fit"]; D0 = fit["D"]; sim0 = (D0.src == "sim").to_numpy(); pan0 = D0.panel.to_numpy()
            xf = pd.Series(fit["eta"][sim0], index=D0.key[sim0].astype(np.int64).to_numpy())
            ms = src == "sim"
            row = {"target": tg, "alpha": a, "lambda": fit["lam"], "auc": fit["auc"], "n": len(w),
                   "ess": sum(per.values()), "ess_ratio": sum(per.values()) / len(w),
                   "min_panel_ratio": f"{pmin} {per[pmin]:.0f}/{npn[pmin]}", "clipped": d["clipped"],
                   "w_max": float(w.max()), "w_q99": float(np.quantile(w, 0.99)), "w_q01": float(np.quantile(w, 0.01)),
                   "corr_eta_full_vs_crossfit_sim": float(np.corrcoef(
                       d["eta"][ms], xf.reindex(T.key[ms].astype(np.int64).to_numpy()).to_numpy())[0, 1])}
            for s in ("cand", "sim", "off"):
                m = src == s
                row[f"ess_ratio_{s}"] = sum(_kish(w[m & (pan == p)]) for p in PANELS8) / m.sum()
                row[f"mean_w_{s}"] = float(w[m].mean())
            XT = d["XT"]; X0 = fit["X"]; tgm = (D0.src == tg).to_numpy()
            for c in BAL_COLS:
                x = XT[c].to_numpy(np.float64); x0 = X0[c].to_numpy(np.float64)
                s_ = np.sqrt(np.mean([np.var(x0[sim0 & (pan0 == p)]) for p in PANELS8])) or 1.0
                d0 = d1 = 0.0
                for p in PANELS8:
                    mp = pan == p; mt = tgm & (pan0 == p)
                    d0 += (x0[mt].mean() - x[mp].mean()) / len(PANELS8)
                    d1 += (x0[mt].mean() - (w[mp] * x[mp]).sum() / w[mp].sum()) / len(PANELS8)
                row[f"smd_{c}"] = f"{d0 / s_:+.2f} > {d1 / s_:+.2f}"
            rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# weighted stage-1 OOF and stage 2
def oof_path(target: str, alpha: float, variant: str, seed: int) -> Path:
    return OUT / f"oof_queue_onset_iw_{target}_{atag(alpha)}_{variant}_s{seed}.parquet"


def mean_path(target: str, alpha: float) -> Path:
    return OUT / f"oof_queue_onset_iw_{target}_{atag(alpha)}_mean9.parquet"


def stack_path(target: str, alpha: float) -> Path:
    return OUT / f"stack_oof_iw_{target}_{atag(alpha)}.parquet"


def run_oof(variant: str, seed: int, target: str, alpha: float = 0.5) -> Path:
    """``robust cv VARIANT p1 --cond queue_onset --oprior`` with T2_OOF_ALL=1 and T2_SEED=seed, with every row
    weighted by its window's importance weight toward ``target``. Rows equal the v6 OOF rows."""
    from .cv import gather, oof
    from .models import CFG
    from .oprior import add_to_rows
    from .robust import VARIANTS
    out = oof_path(target, alpha, variant, seed)
    assert not out.exists(), f"{out} exists"
    t = time.time()
    W = train_weights(target, alpha)
    params, rounds = CFG["p1"]
    R, M = gather(COND, cand_frac=1.0, drop=VARIANTS[variant])
    R = add_to_rows(R, M, PANELS8)
    sw = W.reindex(R.gw).to_numpy(np.float32)
    O = pd.DataFrame(dict(gw=R.gw, k=R.k, link=R.link, y=R.y))
    O["p"] = oof(R, {**params, "seed": seed}, rounds, pred_all=True, sample_weight=sw)
    del R, M
    gc.collect()
    ref = pd.read_parquet(T2H / "oof_queue_onset_rob_on_v3_p1_op_all_new.parquet", columns=["gw", "k", "link", "y"])
    assert len(ref) == len(O) and (ref.to_numpy() == O[["gw", "k", "link", "y"]].to_numpy()).all(), "rows differ from v6"
    O.to_parquet(out)
    print(f"oof {out.name}: {len(O)} rows, row-weight mean {sw.mean():.4f} max {sw.max():.2f}, {time.time() - t:.0f}s",
          flush=True)
    return out


def weighted_mean_oof(target: str, alpha: float) -> Path:
    """0.75 / 0.25 weighted mean of the nine weighted OOF files (stack_v8.mean_oof, written to OUT)."""
    out = mean_path(target, alpha)
    if out.exists():
        return out
    B = None
    for v, s, w in SPECS:
        x = pd.read_parquet(oof_path(target, alpha, v, s))
        if B is None:
            B = x[["gw", "k", "link", "y"]].copy(); B["p"] = 0.0
        assert (x[["gw", "k", "link"]].to_numpy() == B[["gw", "k", "link"]].to_numpy()).all(), (v, s)
        B["p"] += w * x.p.to_numpy(np.float64)
    B["p"] = (B.p / sum(w for *_, w in SPECS)).astype(np.float32)
    B.to_parquet(out)
    return out


def _stage2_frame(target: str, alpha: float):
    from . import stack
    stack.EXTRA = False
    X, cols = stack.build([str(weighted_mean_oof(target, alpha))])
    wr = train_weights(target, alpha).reindex(X.gw.to_numpy()).to_numpy(np.float32)
    assert np.isfinite(wr).all()
    return X, cols, wr


def run_stack(target: str, alpha: float = 0.5) -> Path:
    """Weighted stage-2 OOF (4 week folds, 3 seeds, stack.P2, 300 rounds) on the weighted seeds9 mean."""
    from . import stack
    out = stack_path(target, alpha)
    assert not out.exists(), f"{out} exists"
    t = time.time()
    X, cols, wr = _stage2_frame(target, alpha)
    p2 = np.zeros(len(X))
    for s in STACK_SEEDS:
        p2 += stack.cv(X, cols, {**stack.P2, "seed": s}, 300, weight=wr)
    X["p1"] = X["s_p"]
    X["p2"] = p2 / len(STACK_SEEDS)
    X[["gw", "link", "y", "fold", "src", "p1", "p2"]].to_parquet(out)
    print(f"stack {out.name}: {len(cols)} features, {time.time() - t:.0f}s", flush=True)
    return out


# ----------------------------------------------------------------------------
# evaluation
def compare_row(E, df: pd.DataFrame, ref: pd.DataFrame, name: str) -> dict:
    """Plain CV (hybrid / old truth: Δ sim ± paired SE, Δ off, Δ recurrence slices ± SE, better / worse) and
    the validation- and private-weighted Δ ± SE (shift_cv.score) of a window_iou frame against ``ref``."""
    c = E.compare(df, ref)
    r = {"variant": name}
    for t in E.truths:
        x = c[t]
        r[f"plain_{t}"] = x["d_sim"]; r[f"plain_{t}_se"] = x["se_sim"]; r[f"off_{t}"] = x["d_off"]
        r[f"rec05_{t}"] = x["d_recur<0.05"]; r[f"rec05_{t}_se"] = x["se_recur<0.05"]
        r[f"rec20_{t}"] = x["d_recur<0.2"]; r[f"rec20_{t}_se"] = x["se_recur<0.2"]
        r[f"bw_{t}"] = f"{x['better']}/{x['worse']}"
    for tg in TARGETS:
        s = sc.score(df, COND, tg, ref=ref, cols=[f"iou_{t}" for t in E.truths])
        for t in E.truths:
            r[f"{tg[:3]}_{t}"] = s[f"iou_{t}"]["est"]; r[f"{tg[:3]}_{t}_se"] = s[f"iou_{t}"]["se"]
    return r


def cheap(alphas=(0.5, 1.0)) -> pd.DataFrame:
    """Step 2: one weighted stage-1 component (on_v3 seed 0) per target vs the unweighted seed 0, next to the
    seed noise (unweighted seeds 1-5 vs seed 0 on the same metrics)."""
    from .stack_v8 import oof_name
    E = sc.onset_eval()
    base = {s: E.window_iou(E.load(oof_name("on_v3", s))) for s in range(6)}
    rows = []
    for a in alphas:
        for tg in TARGETS:
            f = oof_path(tg, a, "on_v3", 0)
            if f.exists():
                rows.append(compare_row(E, E.window_iou(E.load(str(f))), base[0], f"iw {tg} {atag(a)} on_v3 s0"))
    for s in range(1, 6):
        rows.append(compare_row(E, base[s], base[0], f"null: unweighted on_v3 s{s}"))
    T = pd.DataFrame(rows).set_index("variant")
    num = [c for c in T.columns if not c.startswith("bw_")]
    null = T.loc[T.index.str.startswith("null"), num]
    T.loc["seed-noise sd (5 null Δs)"] = null.std(ddof=1)
    T.loc["mean of 6 unweighted seeds - s0"] = null.sum() / 6
    return T


def cheap_decision(T: pd.DataFrame, alpha: float = 0.5) -> dict:
    """Pre-specified rule (plan): promising if the validation model's validation-weighted hybrid Δ > 0 and
    > max(seed-noise sd, 0.001), its plain hybrid Δ >= -0.002, and the private model is not below -1 seed-noise
    sd on the private weighting."""
    sd = T.loc["seed-noise sd (5 null Δs)"]
    out = {}
    v, p = f"iw validation {atag(alpha)} on_v3 s0", f"iw private {atag(alpha)} on_v3 s0"
    if v in T.index:
        r = T.loc[v]
        out["validation"] = bool(r.val_hybrid > max(sd.val_hybrid, 0.001) and r.plain_hybrid >= -0.002)
    if p in T.index:
        out["private_not_worse"] = bool(T.loc[p].pri_hybrid > -sd.pri_hybrid)
    out["promising"] = bool(out.get("validation", False) and out.get("private_not_worse", False))
    return out


def variant_probs(E, target: str, alpha: float) -> dict:
    X = pd.read_parquet(stack_path(target, alpha))
    X["k"] = K
    p1, p2 = E.align(X, "p1"), E.align(X, "p2")
    assert np.allclose(p1, E.load(str(mean_path(target, alpha))), atol=1e-6), "stage-1 mean mismatch"
    return {"stage1": p1, "final": STACK_W * p2 + (1 - STACK_W) * p1}


def full_table(alpha: float = 0.5, n_refit: int = 200) -> tuple[pd.DataFrame, dict]:
    """Step 3: each target's weighted v8 recipe (seeds9 + stack 0.3) vs v8, and its weighted stage 1 vs v8's
    stage 1; own-target SE with the weight-model refit; weighted levels."""
    E = sc.onset_eval()
    P = sc.onset_probs()
    ref, ref1 = E.window_iou(P["G2"]), E.window_iou(P["seeds9"])
    rows, out = [], {}
    for tg in TARGETS:
        if not stack_path(tg, alpha).exists():
            continue
        V = variant_probs(E, tg, alpha)
        df = E.window_iou(V["final"])
        out[tg] = (V["final"], df)
        r = compare_row(E, df, ref, f"iw {tg} {atag(alpha)}: seeds9+stack03 vs v8")
        r[f"{tg[:3]}_hybrid_se_refit"] = sc.boot_refit(df, ref, "iou_hybrid", COND, tg, n_boot=n_refit)["se"]
        for w_ in TARGETS:
            for t in E.truths:
                r[f"level_{w_[:3]}_{t}"] = sc.weighted_agg(df, sc.weights(COND, w_), f"iou_{t}")
                r[f"level_{w_[:3]}_{t}_v8"] = sc.weighted_agg(ref, sc.weights(COND, w_), f"iou_{t}")
        rows.append(r)
        rows.append(compare_row(E, E.window_iou(V["stage1"]), ref1, f"iw {tg} {atag(alpha)}: stage 1 vs v8 stage 1"))
    return pd.DataFrame(rows).set_index("variant"), out


def gate12(r: pd.Series, target: str) -> dict:
    """Gate 1 (plain CV) and gate 2 (own-target weighted CV) for one target model's row of ``full_table``."""
    g1 = bool(r.plain_hybrid >= -0.002 and r.plain_old >= -0.002 and r.plain_old >= -r.plain_old_se
              and (r.plain_hybrid > 0 or (r.rec05_hybrid > 0 and r.rec20_hybrid > 0)))
    est, se = r[f"{target[:3]}_hybrid"], r[f"{target[:3]}_hybrid_se"]
    return {"G1_plain": g1, "G2_weighted": bool(est > 0 and est >= -se)}


# ----------------------------------------------------------------------------
# full-train models, validation / private probabilities, footprint, candidate
def model_path(target: str, alpha: float, variant: str, seed: int) -> Path:
    return OUT / f"model_iw_{target}_{atag(alpha)}_{variant}_s{seed}_queue_onset.txt"


def stage1_model(target: str, alpha: float, variant: str, seed: int):
    import lightgbm as lgb
    from .robust_pipeline import train_variant
    f = model_path(target, alpha, variant, seed)
    if f.exists():
        return lgb.Booster(model_file=str(f))
    m = train_variant(variant, "p1", False, cond=COND, seed=seed, oprior=True, gw_weight=train_weights(target, alpha))
    m.save_model(str(f))
    return m


def stage2_full(target: str, alpha: float):
    """Weighted stage-2 models on every onset window (as onset_v8.stage2_models)."""
    import lightgbm as lgb
    from . import stack
    X, cols, wr = _stage2_frame(target, alpha)
    ds = lgb.Dataset(X[cols].to_numpy(np.float32), X.y.to_numpy(np.float32), weight=wr, feature_name=cols).construct()
    del X
    gc.collect()
    return [lgb.train({**stack.P2, "seed": s}, ds, 300) for s in STACK_SEEDS], cols


def target_probs(target: str, alpha: float) -> pd.DataFrame:
    """The target month's onset rows predicted by its weighted models: p1 (stage-1 weighted mean), p2, p."""
    from .onset_v8 import stage2_probs
    from .robust_pipeline import probs
    out = OUT / f"probs_iw_{target}_{atag(alpha)}.parquet"
    if out.exists():
        return pd.read_parquet(out)
    t = time.time()
    parts = []
    for v, s, _ in SPECS:
        m = stage1_model(target, alpha, v, s)
        parts.append(probs(m, target, COND))
        del m
        gc.collect()
        print(f"  stage 1 {target} {v} s{s}: {time.time() - t:.0f}s", flush=True)
    B = parts[0][KEY].copy()
    for q in parts[1:]:
        assert (q[KEY].to_numpy() == B[KEY].to_numpy()).all()
    wt = np.array([w for *_, w in SPECS])
    B["p1"] = np.tensordot(wt / wt.sum(), np.stack([q.p.to_numpy() for q in parts]), 1)
    ms, cols = stage2_full(target, alpha)
    B["p2"] = stage2_probs(B[KEY + ["p1"]].rename(columns={"p1": "p"}).reset_index(drop=True), ms, cols)
    B["p"] = STACK_W * B.p2 + (1 - STACK_W) * B.p1
    B["split"] = target
    B.to_parquet(out)
    print(f"probs {out.name}: {len(B)} rows, {time.time() - t:.0f}s", flush=True)
    return B


def _sets_cv(E, p: np.ndarray) -> dict:
    return {w["gw"]: {(K, int(l)) for l in w["link"][eiou_topm(p[w["idx"]])[0]]} for w in E.win}


def _sets_test(P: pd.DataFrame) -> dict:
    out = {}
    for wid, g in P.groupby("window_id"):
        g = g.sort_values(["k", "link"])
        out[wid] = {(K, int(l)) for l in g.link.to_numpy()[eiou_topm(g.p.to_numpy())[0]]}
    return out


def footprint(target: str, p_cv: np.ndarray, P_t: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """shift_cv.footprint_check of the target model against v8: CV edits = its OOF top-m sets vs v8's OOF sets;
    target edits = its sets on the target month vs v8's."""
    E = sc.onset_eval()
    rows = {}
    for A, B in ((_sets_cv(E, p_cv), sc.cv_sets(COND, "G2")), (_sets_test(P_t), sc.test_sets(COND, "G2"))):
        for k in A:
            a, b = A[k], B[k]
            n = len(a ^ b)
            rows[str(k)] = (np.log1p(n), float(n > 0), len(a - b), len(b - a), len(b))
    Ed = pd.DataFrame.from_dict(rows, orient="index", columns=["fp_lchg", "fp_any", "add", "rem", "nref"])
    D = sc.window_features(COND)
    need = D.key[D.src.isin(["sim", target])].astype(str)
    assert need.isin(Ed.index).all(), "edits missing for some windows"
    return sc.footprint_check(COND, Ed, None, target), Ed


def ongoing_lines_identical(a: Path, b: Path) -> bool:
    """Raw text of every ongoing row (and the header) is byte-identical."""
    W = pd.concat([official_windows(p, s) for p in PANELS8 for s in TARGETS]).set_index("window_id")
    la = a.read_bytes().split(b"\n"); lb = b.read_bytes().split(b"\n")
    if len(la) != len(lb) or la[0] != lb[0]:
        return False
    for x, y in zip(la[1:], lb[1:]):
        if not x:
            if y:
                return False
            continue
        if W.condition.get(x.split(b",", 1)[0].decode()) != COND and x != y:
            return False
    return True


def write_onset_rows(preds: dict, out: Path, base: Path = BASE_CSV) -> dict:
    """``base`` with only the onset rows replaced (T+30 only); per-split change counts vs base."""
    from ..data import tindex
    sub = pd.read_csv(base, dtype=str)
    W = pd.concat([official_windows(p, s) for p in PANELS8 for s in TARGETS]).set_index("window_id")
    on = sub.window_id.map(W.condition).to_numpy() == COND
    k = tindex(sub.timestamp) - sub.window_id.map(W["T"]).to_numpy() - 1
    q_old = sub.queue_pred.astype(int).to_numpy()
    q = q_old.copy()
    lids = {p: statics(p)["lid"] for p in PANELS8}
    pan = sub.window_id.map(W.panel).to_numpy()
    for i in np.flatnonzero(on):
        q[i] = int(k[i] == K - 1 and preds[sub.window_id.iat[i]][lids[pan[i]][sub.link_id.iat[i]]])
    sub["queue_pred"] = q
    sub.to_csv(out, index=False)
    res = {"ongoing_rows_changed": int((q != q_old)[~on].sum()),
           "onset_cells_not_at_T30": int(((q == 1) & on & (k != K - 1)).sum())}
    d = pd.DataFrame({"wid": sub.window_id[on].to_numpy(), "link": sub.link_id[on].to_numpy(), "old": q_old[on],
                      "new": q[on]})
    d["split"] = np.where(d.wid.str.contains("_validation_"), "validation", "private")
    for s, g0 in d.groupby("split"):
        g0 = g0.assign(i=g0.old & g0.new, u=g0.old | g0.new)
        per = g0.groupby("wid")[["i", "u", "old", "new"]].sum()
        agree = np.where(per.u > 0, per.i / per.u.clip(lower=1), 1.0)
        chg = g0[g0.old != g0.new]
        res[s] = dict(windows=int(len(per)), windows_changed=int((per.i != per.u).sum()),
                      cells_changed=int(len(chg)), cells_added=int((chg.new == 1).sum()),
                      cells_removed=int((chg.old == 1).sum()), cells_base=int(per.old.sum()),
                      cells_new=int(per.new.sum()), empty_windows=int((per.new == 0).sum()),
                      cells_per_window_min=int(per.new.min()), cells_per_window_max=int(per.new.max()),
                      mean_window_agreement=round(float(agree.mean()), 4),
                      min_window_agreement=round(float(agree.min()), 4),
                      changes={w: {"n": f"{int(per.old[w])}->{int(per.new[w])}",
                                   "added": g.link[g.new == 1].tolist(), "removed": g.link[g.old == 1].tolist()}
                               for w, g in chg.groupby("wid")})
    return res


def write_candidate(name: str, P: pd.DataFrame) -> dict:
    """lgb_v11_<name>.csv (v8 file, onset rows replaced) + probs_v11_<name>_onset.parquet (v8 layout / order)."""
    from .onset_v8 import decode
    from .submit import check
    out = CSV_DIR / f"lgb_v11_{name}.csv"
    pf = T2H / f"probs_v11_{name}_onset.parquet"
    assert not out.exists() and not pf.exists(), "candidate files exist"
    V8 = pd.read_parquet(V8_PROBS)
    kv = V8.window_id.astype(str) + "|" + V8.link.astype(int).astype(str)
    kp = P.window_id.astype(str) + "|" + P.link.astype(int).astype(str)
    pm = pd.Series(P.p.to_numpy(np.float64), index=kp.to_numpy())
    assert pm.index.is_unique and len(pm) == len(V8)
    Q = V8[KEY + ["split"]].copy()
    Q["p"] = pm.reindex(kv.to_numpy()).to_numpy()
    assert Q.p.notna().all()
    Q = Q[KEY + ["p", "split"]]
    Q.to_parquet(pf)
    res = write_onset_rows(decode(Q), out)
    res["ongoing_lines_byte_identical"] = ongoing_lines_identical(BASE_CSV, out)
    res["check"] = check(out)
    res.update(file=str(out), probs=str(pf))
    return res


def build(name: str, alpha: float = 0.5) -> dict:
    """Gates 1-3 per target model; the candidate is written only if every gate passes for both months."""
    Tb, V = full_table(alpha)
    res = {"name": name, "alpha": alpha, "targets": {}}
    Ps = {}
    for tg in TARGETS:
        r = Tb.loc[f"iw {tg} {atag(alpha)}: seeds9+stack03 vs v8"]
        g = gate12(r, tg)
        Ps[tg] = target_probs(tg, alpha)
        fc, Ed = footprint(tg, V[tg][0], Ps[tg])
        g["G3_footprint"] = bool(all(v["pct"] < sc.FP_FLAG for v in fc.values()))
        te = Ed.loc[[k for k in Ed.index if f"_{tg}_" in k]]
        res["targets"][tg] = {"gate": g, "footprint": fc, "edited_windows": int((te.fp_any > 0).sum()),
                              "cells_added": int(te["add"].sum()), "cells_removed": int(te.rem.sum())}
        print(tg, json.dumps(res["targets"][tg], default=float), flush=True)
    res["pass"] = bool(all(all(v["gate"].values()) for v in res["targets"].values()))
    if res["pass"]:
        res["candidate"] = write_candidate(name, pd.concat([Ps[t] for t in TARGETS], ignore_index=True))
    (OUT / f"build_{name}.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({k: v for k, v in res.items() if k != "targets"}, default=float), flush=True)
    return res


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["weights", "cv", "cheap", "stack", "table", "build"])
    ap.add_argument("args", nargs="*")
    ap.add_argument("--alpha", type=float, default=0.5)
    a = ap.parse_args()
    _env_ok()
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 80); pd.set_option("display.max_rows", 200)
    t = time.time()
    if a.cmd == "weights":
        R = weight_report()
        R.to_csv(OUT / "weights_report.csv", index=False)
        print(R.round(4).T.to_string(), flush=True)
        for tg in TARGETS:
            W = train_weights(tg, a.alpha)
            T = train_window_features().set_index("key").reindex(W.index)
            pd.DataFrame({"gw": W.index, "panel": T.panel.to_numpy(), "src": T.src.to_numpy(), "w": W.to_numpy()}
                         ).to_parquet(OUT / f"iw_{tg}_{atag(a.alpha)}.parquet")
    elif a.cmd == "cv":
        run_oof(a.args[0], int(a.args[1]), a.args[2], a.alpha)
    elif a.cmd == "cheap":
        T = cheap()
        T.to_csv(OUT / "cheap_table.csv")
        print(T.round(4).T.to_string(), flush=True)
        print("decision", cheap_decision(T, a.alpha), flush=True)
    elif a.cmd == "stack":
        run_stack(a.args[0], a.alpha)
    elif a.cmd == "table":
        T, _ = full_table(a.alpha)
        T.to_csv(OUT / f"full_table_{atag(a.alpha)}.csv")
        print(T.round(4).T.to_string(), flush=True)
        for tg in TARGETS:
            name = f"iw {tg} {atag(a.alpha)}: seeds9+stack03 vs v8"
            if name in T.index:
                print(tg, "gates 1-2", gate12(T.loc[name], tg), flush=True)
    elif a.cmd == "build":
        build(a.args[0], a.alpha)
    print(f"{a.cmd} done in {time.time() - t:.0f}s, peak_rss_kb {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}",
          flush=True)


if __name__ == "__main__":
    main()
