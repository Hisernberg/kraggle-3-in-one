"""Holdout J of per-kind weighted Task 1 ensembles (regular and blackout cells weighted separately).

The ramp-feature member (hold7) is much better on blackout ("dark") cells than the members without
ramp features (dark speed RMSE 6.11 vs 6.89), so an equal-weight average dilutes it. This scores
weightings where the ramp member gets weight w_reg on regular cells and w_dark on blackout cells and
the other members share the rest equally, with the adopted post-processing (gate 0.6, a 0.75, TV).

    python -m trafficflow.t1_weighted_eval hold7 hold3,hold4,hold5
    python -m trafficflow.t1_weighted_eval kinds hold3,hold4,hold5,hold7 "hold7;hold8;hold7+hold8"
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from .t1_pipeline import WORK
from .t1_smooth import DEFAULT
from .t1_smooth_eval import CH3, HOLD_PANELS, Hold, load_preds

SCHEMES = {"base (others only)": (0.0, 0.0), "equal": None, "dark 0.5": ("eq", 0.5), "dark 0.75": ("eq", 0.75),
           "dark 1.0": ("eq", 1.0), "reg 0.5 / dark 0.75": (0.5, 0.75), "reg 0.5 / dark 1.0": (0.5, 1.0)}


def combine(new: dict, others: list, dark: np.ndarray, w_reg: float, w_dark: float) -> dict:
    """new gets w_reg on regular rows and w_dark on dark rows; the others share the rest equally."""
    w = np.where(dark, w_dark, w_reg)
    rest = np.mean([o["speed"] for o in others], 0), np.mean([o["flow"] for o in others], 0), \
        np.mean([o["dens"] for o in others], 0)
    return {c: w * new[c] + (1 - w) * r for c, r in zip(CH3, rest)}


def run(new_tag: str, other_tags: list[str], panels=HOLD_PANELS) -> pd.DataFrame:
    rows = []
    n = len(other_tags) + 1
    for p in panels:
        H = Hold(p)
        new = load_preds(p, new_tag)
        others = [load_preds(p, t) for t in other_tags]
        dark = np.asarray(H.dark_row, bool)[H.r]      # blackout flag per slot -> per cell
        for name, sc in SCHEMES.items():
            if sc is None:
                w_reg = w_dark = 1.0 / n
            else:
                w_reg = 1.0 / n if sc[0] == "eq" else sc[0]
                w_dark = sc[1]
            H.pr = combine(new, others, dark, w_reg, w_dark)
            v, q, g = H.base()
            v, q = H.smooth(v, q, g, **DEFAULT)
            s = H.score(v, q)
            rows.append(dict(panel=p, scheme=name, **{k: s[k] for k in ("J", "S_state", "LWR")}))
            print(p, name, {k: round(s[k], 5) for k in ("J", "S_state", "LWR")}, flush=True)
    d = pd.DataFrame(rows)
    piv = d.pivot_table(index="scheme", columns="panel", values="J").reindex(list(SCHEMES))
    base = piv.loc["base (others only)"]
    out = piv.assign(mean=piv.mean(axis=1), dJ_vs_base=(piv - base).mean(axis=1), up=(piv > base).sum(axis=1))
    print(out.round(5).to_string())
    return d


def run_kinds(reg_tags: list[str], dark_sets: list[list[str]], panels=HOLD_PANELS) -> pd.DataFrame:
    """Regular cells = mean of `reg_tags`; blackout cells = mean of each set in `dark_sets` (the first set is
    the reference). A dark-only member (TFB_KINDS=dark, NaN on regular cells) can only be in `dark_sets`."""
    rows = []
    for p in panels:
        H = Hold(p)
        P = {t: load_preds(p, t) for t in dict.fromkeys(reg_tags + [t for s in dark_sets for t in s])}
        dark = np.asarray(H.dark_row, bool)[H.r]
        reg = {c: np.mean([P[t][c] for t in reg_tags], 0) for c in CH3}
        for ds in dark_sets:
            dk = {c: np.mean([P[t][c] for t in ds], 0) for c in CH3}
            H.pr = {c: np.where(dark, dk[c], reg[c]) for c in CH3}
            assert all(np.isfinite(H.pr[c]).all() for c in CH3), f"{p} {ds}: non-finite predictions"
            v, q, g = H.base()
            v, q = H.smooth(v, q, g, **DEFAULT)
            s = H.score(v, q)
            e = {c: float(np.sqrt(np.mean((dk[c][dark] - t[dark]) ** 2)))
                 for c, t in (("speed", H.ys), ("flow", H.yq))}
            name = "+".join(ds)
            rows.append(dict(panel=p, dark=name, **{k: s[k] for k in ("J", "S_state", "LWR")},
                             dark_rmse_v=e["speed"], dark_rmse_q=e["flow"]))
            print(p, name, {k: round(s[k], 5) for k in ("J", "S_state", "LWR")},
                  {k: round(x, 3) for k, x in e.items()}, flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(WORK / "smooth" / "kinds.csv", index=False)
    order = ["+".join(s) for s in dark_sets]
    piv = d.pivot_table(index="dark", columns="panel", values="J").reindex(order)
    base = piv.iloc[0]
    out = piv.assign(mean=piv.mean(axis=1), dJ=(piv - base).mean(axis=1), up=(piv > base).sum(axis=1))
    print(out.round(5).to_string())
    print(d.pivot_table(index="dark", columns="panel", values="dark_rmse_v").reindex(order).round(3).to_string())
    return d


def run_regs(reg_sets: dict, dark_tags: list[str], panels=HOLD_PANELS) -> pd.DataFrame:
    """Blackout cells = mean of `dark_tags` (fixed); regular cells = each entry of `reg_sets` (name -> list of
    (tag, weight)); the first entry is the reference. A regular-only member (TFB_KINDS=reg, NaN on blackout
    cells) can only appear in reg_sets."""
    rows = []
    for p in panels:
        H = Hold(p)
        tags = dict.fromkeys(dark_tags + [t for v in reg_sets.values() for t, _ in v])
        P = {t: load_preds(p, t) for t in tags}
        dark = np.asarray(H.dark_row, bool)[H.r]
        dk = {c: np.mean([P[t][c] for t in dark_tags], 0) for c in CH3}
        for name, members in reg_sets.items():
            ws = sum(w for _, w in members)
            rg = {c: sum(w * np.nan_to_num(P[t][c]) for t, w in members) / ws for c in CH3}
            H.pr = {c: np.where(dark, dk[c], rg[c]) for c in CH3}
            assert all(np.isfinite(H.pr[c][~dark]).all() for c in CH3), f"{p} {name}: non-finite regular predictions"
            v, q, g = H.base()
            v, q = H.smooth(v, q, g, **DEFAULT)
            s = H.score(v, q)
            e = float(np.sqrt(np.mean((rg["speed"][~dark] - H.ys[~dark]) ** 2)))
            rows.append(dict(panel=p, reg=name, **{k: s[k] for k in ("J", "S_state", "LWR")}, reg_rmse_v=e))
            print(p, name, {k: round(s[k], 5) for k in ("J", "S_state", "LWR")}, round(e, 4), flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(WORK / "smooth" / "regs.csv", index=False)
    order = list(reg_sets)
    piv = d.pivot_table(index="reg", columns="panel", values="J").reindex(order)
    base = piv.iloc[0]
    print(piv.assign(mean=piv.mean(axis=1), dJ=(piv - base).mean(axis=1), up=(piv > base).sum(axis=1)).round(5).to_string())
    print(d.pivot_table(index="reg", columns="panel", values="reg_rmse_v").reindex(order).round(4).to_string())
    return d


if __name__ == "__main__":
    if sys.argv[1] == "regspec":  # regspec '<json {name: [[tag, w], ...]}; first = reference>' <dark tags,comma>
        import json
        spec = json.loads(sys.argv[2])
        run_regs({k: [(t, float(w)) for t, w in v] for k, v in spec.items()}, sys.argv[3].split(","))
    elif sys.argv[1] == "regs":  # regs <new tag> <existing reg tags,comma> <dark tags,comma>
        new, old, dks = sys.argv[2], sys.argv[3].split(","), sys.argv[4].split(",")
        n = len(old)
        run_regs({"base": [(t, 1.0) for t in old], "add": [(t, 1.0) for t in old] + [(new, 1.0)],
                  "half": [(t, 0.5 / n) for t in old] + [(new, 0.5)], "alone": [(new, 1.0)]}, dks)
    elif sys.argv[1] == "kinds":  # kinds <reg tags,comma> "<dark set;dark set;...>" (a set: tag+tag)
        run_kinds(sys.argv[2].split(","), [s.split("+") for s in sys.argv[3].split(";")])
    else:
        run(sys.argv[1], sys.argv[2].split(","))
