"""Blackout-row smoothing strength checked on the simulated test-month blackouts (``t1_pseudo.py dark``).

Every evaluation blackout blanks rows T+1..T+18 of a March/April day; rows T and T+19 stay observed. The dark model's
predictions of the eligible cells in the span go through the production post-processing (gated reconciliation
a = 0.75 below 0.6 v_f, then ``t1_smooth.smooth_cells`` with the blackout category), and are scored with
- the LWR proxy restricted to the span: sum |dN_sub - dN_true| / sum |dN_true| over consecutive known cells of a
  link from T to T+19 (N = per-lane density x lanes x length; the observed rows T, T+19 carry the truth), and
- the blackout-cell RMSE of speed and flow per lane (the S_state part).

    python -m trafficflow.t1_dsmooth fullPD:0.5,fullPD2:0.5 0,0.025,0.05,0.1,0.2
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from .data import PANELS, build_panel, network
from .t1 import mileposts
from .t1_holdout import reconcile
from .t1_pipeline import WORK
from .t1_smooth import smooth_cells

OUT = WORK / "pseudo"


def panel_eval(panel: str, members, taus, tau_a: float = 0.0, a_out: float = 0.0) -> list[dict]:
    X = pd.read_parquet(OUT / f"{panel}_dark.parquet", columns=["t", "j", "split", "origin", "y_speed", "y_flow", "lanes_", "vf_"])
    X = X[X.split.str.endswith("_eval")].reset_index(drop=True)
    if not len(X):
        return []
    P = {t: dict(np.load(OUT / f"dpred_{t}_{panel}.npz")) for t, _ in members}
    pr = {c: sum(w * P[t][c] for t, w in members) for c in ("speed", "flow", "dens")}
    vf = X.vf_.to_numpy(np.float64)
    v, q = reconcile(pr["speed"], pr["flow"], pr["dens"], 0.75)
    gate = pr["speed"] < 0.6 * vf
    v = np.where(gate, v, pr["speed"]); q = np.where(gate, q, pr["flow"])
    z = np.load(build_panel(panel), allow_pickle=True)
    links = [str(x) for x in z["links"]]
    order = np.argsort(mileposts(panel, links))               # t1.Panel link order -> cache column
    col = order[X.j.to_numpy().astype(np.int64)]
    net = network(panel)["fd"]
    lanes = net.lanes.to_numpy(np.float64); length = net.length_km.to_numpy(np.float64)
    t = X.t.to_numpy().astype(np.int64)
    t0, t1 = int(t.min()) - 1, int(t.max()) + 2
    xs = z["speed"][t0:t1].astype(np.float64); xf = z["flow"][t0:t1].astype(np.float64)
    tg = z["target"][t0:t1]
    obs = np.isfinite(xs) & np.isfinite(xf) & (tg == 0)
    origins = X.origin.unique()
    for T in origins:                                         # the simulated blackout hides these rows
        obs[T + 1 - t0:T + 19 - t0] = False
    kobs = np.where(obs, xf / lanes[None, :] / np.maximum(xs, 1), np.nan)
    ktrue = X.y_flow.to_numpy() / np.maximum(X.y_speed.to_numpy(), 1)
    from .data import SLOTS, SPLIT_DAYS
    base = {}
    for split in ("validation", "private"):
        a, b = SPLIT_DAYS[split]
        ys = z["speed"][a * SLOTS:b * SLOTS].astype(np.float64); yf = z["flow"][a * SLOTS:b * SLOTS].astype(np.float64)
        N = yf / np.maximum(ys, 1) * length[None, :]              # total vehicles (flow is total, not per lane)
        dN = np.abs(np.diff(N, axis=0))
        base[split] = (float(np.nanmean(dN)), int(np.isfinite(dN).sum()), int(dN.size))
    rows = []
    for tau in taus:
        if tau > 0 or tau_a > 0:
            vs, qs = smooth_cells(v, q, t - t0, col, xs.shape, kobs, gate, np.ones(len(X), bool), dark=tau, dark_a=tau_a,
                                  a_out=a_out)
        else:
            vs, qs = v, q
        ks = qs / np.maximum(vs, 1)
        for split in ("validation_eval", "private_eval"):
            m = (X.split == split).to_numpy()
            if not m.any():
                continue
            num = den = nb = 0.0
            for T in np.unique(X.origin.to_numpy()[m]):
                mm = m & (X.origin.to_numpy() == T)
                Ns = np.full((20, len(links)), np.nan); Nt = np.full((20, len(links)), np.nan)
                r = t[mm] - T
                Ns[r, col[mm]] = ks[mm]; Nt[r, col[mm]] = ktrue[mm]
                for rr in (0, 19):                            # observed boundary rows: truth on both sides
                    kb = kobs[T + rr - t0]
                    Ns[rr] = kb; Nt[rr] = kb
                scale = lanes * length
                Ns *= scale[None, :]; Nt *= scale[None, :]
                ok = np.isfinite(Nt[1:]) & np.isfinite(Nt[:-1])
                err = np.abs((Ns[1:] - Ns[:-1]) - (Nt[1:] - Nt[:-1]))
                num += err[ok].sum()
                den += np.abs(Nt[1:] - Nt[:-1])[ok].sum()
                nb += err[[0, -1]][ok[[0, -1]]].sum()            # the two boundary transitions (T->T+1, T+18->T+19)
            bm, bn, bsize = base[split.split("_")[0]]
            rows.append(dict(panel=panel, split=split, tau=tau, num=num, den=den, num_bnd=nb, n=int(m.sum()), n_orig=len(np.unique(X.origin.to_numpy()[m])),
                             base_mean_dN=bm, base_pairs=bsize,
                             sse_v=float(((vs[m] - X.y_speed.to_numpy()[m]) ** 2).sum()),
                             sse_q=float(((qs[m] - X.y_flow.to_numpy()[m]) ** 2).sum())))
    return rows


def run(members, taus, tau_a: float = 0.0, a_out: float = 0.0) -> pd.DataFrame:
    rows = []
    for p in PANELS:
        rows += panel_eval(p, members, taus, tau_a, a_out)
        print(p, "done", flush=True)
    d = pd.DataFrame(rows)
    g = d.groupby(["split", "tau"])[["num", "den", "num_bnd", "n", "sse_v", "sse_q"]].sum()
    g["e_lwr"] = g.num / g.den
    g["bnd_share"] = g.num_bnd / g.num
    g["rmse_v"] = np.sqrt(g.sse_v / g.n); g["rmse_q"] = np.sqrt(g.sse_q / g.n)
    print(g[["e_lwr", "bnd_share", "rmse_v", "rmse_q"]].round(5).to_string())
    d.to_csv(OUT / "dsmooth.csv", index=False)
    return d


if __name__ == "__main__":
    mem = [(m.split(":")[0], float(m.split(":")[1])) for m in sys.argv[1].split(",")]
    taus = [float(x) for x in sys.argv[2].split(",")]
    run(mem, taus, float(sys.argv[3]) if len(sys.argv) > 3 else 0.0, float(sys.argv[4]) if len(sys.argv) > 4 else 0.0)
