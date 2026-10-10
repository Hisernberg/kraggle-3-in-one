"""Month-specific speed-level correction for Task 1 regular rows, learned on the test-month pseudo-holdout.

The pseudo-holdout cells (``t1_pseudo.py``: observed, non-target cells of March/April, hidden and predicted like a
target) show a per-(link, month) speed bias in the transductive members that no feature can express (the members pool
both test months with train). Correction per (panel, link, month, speed band of the raw prediction):

    c = mean(log y_speed - log v) * n / (n + K),   then  v <- v * exp(c),  q <- q * exp(c)

Speed and flow are scaled together, so q/v (the density the LWR score sees) and the gated density reconciliation are
unchanged; only the S_state part moves. Scaling speed alone lowers speed RMSE a little more but raises the density
error by 0.2-0.3%, which costs about as much on S_LWR. Dark (blackout) rows are left as they are.

    python -m trafficflow.t1_bias check  fullP2:0.5,fullP3:0.5       # cross-fitted gain (fold = day parity)
    python -m trafficflow.t1_bias apply  fullP2:0.5,fullP3:0.5 ens_H13P ens_H13Pb   # fit on all cells, write state
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .data import PANELS, SLOTS, SPLIT_DAYS
from .t1_holdout import reconcile
from .t1_pipeline import WORK
from .t1_pseudo import s_state

OUT = WORK / "pseudo"
K = 30.0
EDGES = [0.5, 0.8, 0.95, 1.05]          # raw v / v_f bands
GATE, A = 0.6, 0.75                     # the production reconciliation (make_submission --gate 0.6 --recon-a 0.75)


def _members(spec: str):
    return [(m.split(":")[0], float(m.split(":")[1])) for m in spec.split(",")]


def _production(v0, q0, k0, vf, lanes):
    v, q = reconcile(v0, q0, k0, A)
    g = v0 < GATE * vf
    v = np.where(g, v, v0); q = np.where(g, q, q0)
    return np.clip(v, 3.0, 130.0), np.maximum(q * lanes, 60.0) / lanes


def pseudo_frame(members) -> pd.DataFrame:
    parts = []
    for panel in PANELS:
        X = pd.read_parquet(OUT / f"{panel}.parquet",
                            columns=["t", "j", "split", "regime_day", "y_speed", "y_flow", "lanes_", "vf_"])
        P = {t: dict(np.load(OUT / f"pred_{t}_{panel}.npz")) for t, _ in members}
        for c in ("speed", "flow", "dens"):
            X[f"p_{c}"] = sum(w * P[t][c] for t, w in members)
        X["panel"] = panel
        parts.append(X)
    X = pd.concat(parts, ignore_index=True)
    X["band"] = np.digitize(X.p_speed / X.vf_, EDGES)
    X["v"], X["q"] = _production(X.p_speed.to_numpy(), X.p_flow.to_numpy(), X.p_dens.to_numpy(), X.vf_.to_numpy(),
                                 X.lanes_.to_numpy())
    X["r"] = np.log(np.maximum(X.y_speed, 1.0)) - np.log(X.v)
    return X


KEYS = ["panel", "j", "split", "band"]


def fit(X: pd.DataFrame) -> pd.DataFrame:
    G = X.groupby(KEYS).agg(n=("r", "size"), c=("r", "mean")).reset_index()
    G["c"] *= G.n / (G.n + K)
    return G


def check(members) -> pd.DataFrame:
    X = pseudo_frame(members)
    fold = (X.t // SLOTS) % 2
    c = np.zeros(len(X))
    for f in (0, 1):
        G = fit(X[fold != f])
        c[(fold == f).to_numpy()] = X.loc[fold == f, KEYS].merge(G, on=KEYS, how="left").c.fillna(0).to_numpy()
    rows = []
    for (panel, split), g in X.groupby(["panel", "split"]):
        i = g.index.to_numpy()
        e = np.exp(c[i])
        args = (g.y_speed.to_numpy(), g.y_flow.to_numpy(), g.regime_day.to_numpy())
        rows.append(dict(panel=panel, split=split, dS=s_state(g.v.to_numpy() * e, g.q.to_numpy() * e, *args)
                         - s_state(g.v.to_numpy(), g.q.to_numpy(), *args)))
    d = pd.DataFrame(rows)
    piv = d.pivot_table(index="panel", columns="split", values="dS")
    print(piv.round(5).to_string())
    print("mean dS", piv.mean().round(5).to_dict(), "-> 0.35*dS", (0.35 * piv.mean()).round(5).to_dict(),
          "panels up", (piv > 0).sum().to_dict())
    return d


def _split_of_t(t: np.ndarray) -> np.ndarray:
    out = np.full(len(t), "", object)
    for s in ("validation", "private"):
        a, b = SPLIT_DAYS[s]
        out[(t >= a * SLOTS) & (t < b * SLOTS)] = s
    return out


def _jmap(panel: str) -> pd.DataFrame:
    """Column j -> link_id. From the feat test table when it exists; otherwise from the Panel's milepost order
    (the order feat_panel and t1_pseudo use), so a restored container does not need the heavy feat stage."""
    f = WORK / "feat" / f"{panel}_test.parquet"
    if f.exists():
        return pd.read_parquet(f, columns=["j", "link_id"]).drop_duplicates().assign(panel=panel)
    from .data import load
    from .t1 import mileposts
    d = load(panel)
    links = np.asarray(d["links"])[np.argsort(mileposts(panel, d["links"]))]
    return pd.DataFrame({"j": np.arange(len(links)), "link_id": links, "panel": panel})


def apply(members, tag_in: str, tag_out: str) -> None:
    from .make_submission import fd_per_cell
    G = fit(pseudo_frame(members))
    jmap = pd.concat([_jmap(p) for p in PANELS])
    G = G.merge(jmap, on=["panel", "j"], how="left")
    assert G.link_id.notna().all()
    pr = pd.read_parquet(WORK / "pred" / f"state_{tag_in}.parquet")
    vf, _, _ = fd_per_cell(pr)
    pr["split"] = _split_of_t(pr.t.to_numpy())
    pr["band"] = np.digitize(pr.speed.to_numpy() / vf, EDGES)
    c = pr[["panel", "link_id", "split", "band"]].merge(G[["panel", "link_id", "split", "band", "c"]],
                                                        on=["panel", "link_id", "split", "band"], how="left").c
    c = np.where((pr.kind == "reg").to_numpy(), c.fillna(0.0).to_numpy(), 0.0)
    e = np.exp(c)
    pr["speed"] = pr.speed * e
    pr["flow_lane"] = pr.flow_lane * e
    print(f"state_{tag_out}: {int((c != 0).sum())} regular rows corrected, |c| mean {np.abs(c[c != 0]).mean():.4f}, "
          f"max {np.abs(c).max():.4f}; dark rows unchanged")
    pr.drop(columns=["split", "band"]).to_parquet(WORK / "pred" / f"state_{tag_out}.parquet")


if __name__ == "__main__":
    cmd, spec = sys.argv[1], _members(sys.argv[2])
    if cmd == "check":
        check(spec)
    elif cmd == "apply":
        apply(spec, sys.argv[3], sys.argv[4])
