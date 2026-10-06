"""Diagnostic: where are the March onset queues? The masked view is visible again after the 18 dark rows
(T+19..), so the queued links at T+19..T+21 show where the onset queue sits ~65 min after T+30.
Compare our T+30 predictions with that proxy on validation, and calibrate the proxy on train sim windows
(true T+30 truth known). Diagnostic only; nothing here feeds a prediction."""
import numpy as np
import pandas as pd
from trafficflow.data import load
from trafficflow.t2.core import PANELS8, K, statics, official_windows, iou, FAMILY
from trafficflow.t2.calib import load_oof
from trafficflow.t2.robust import meta_index
from trafficflow.t2.models import eiou_topm
from pathlib import Path
WORK = Path("/home/user/work/t2h")
ROWS = (19, 20, 21)


def post_q(sp, vcut, T):
    s = sp[[T + r for r in ROWS]]
    return ((s <= vcut[None, :]) & np.isfinite(s)).any(0)


def near(a, b, tol=1):
    """a (bool [L]) within tol links of any b cell."""
    if not b.any():
        return np.zeros_like(a)
    idx = np.flatnonzero(b)
    j = np.arange(len(a))
    return a & (np.abs(j[:, None] - idx[None, :]).min(1) <= tol)


rows = []
D = {p: load(p) for p in PANELS8}
# ---- validation + private: our v8 predictions
sub = pd.read_csv("/home/user/work/t2/lgb_v8_seeds9_stack03.csv", dtype={"link_id": str})
for split in ("validation", "private"):
    for p in PANELS8:
        st = statics(p); W = official_windows(p, split); W = W[W.condition == "queue_onset"]
        sp = D[p]["speed"]
        s = sub[sub.window_id.isin(W.window_id) & (sub.queue_pred == 1)]
        for w in W.itertuples():
            pred = np.zeros(st["L"], bool)
            li = s[s.window_id == w.window_id].link_id.map(st["lid"]).to_numpy()
            pred[li] = True
            pq = post_q(sp, st["vcut"], w.T)
            vis = np.isfinite(sp[[w.T + r for r in ROWS]]).any(0)
            rows.append(dict(split=split, panel=p, wid=w.window_id, T=w.T, npred=int(pred.sum()), npost=int(pq.sum()),
                             hit=bool((pred & pq).any()), hit1=bool(near(pred, pq, 1).any()), vis=float(vis.mean()),
                             pred_links=np.flatnonzero(pred).tolist(), post_links=np.flatnonzero(pq).tolist()))
# ---- train sim windows: v6 OOF top-m vs hybrid truth, and the proxy
O = load_oof(); Mi = meta_index("queue_onset")
O = O[O.gw.map(Mi.src).isin(["sim"]).to_numpy()].sort_values("gw", kind="stable")
g = O.gw.to_numpy(); cut = np.flatnonzero(np.diff(g)) + 1
DS = {p: np.load(WORK / f"ds_{p}.npz", allow_pickle=True) for p in PANELS8}
kk = O.k.to_numpy().astype(int) - 1; ll = O.link.to_numpy().astype(int); pp = O.p.to_numpy()
for idx in np.split(np.arange(len(O)), cut):
    gw = int(g[idx[0]]); m = Mi.loc[gw]; p = m.panel; w = int(m.w); z = DS[p]; st = statics(p)
    T = int(z["w_T"][w]); y30 = z["y"][w][K - 1].astype(bool)
    sel = eiou_topm(pp[idx])[0]
    P = np.zeros((K, st["L"]), bool); P[kk[idx][sel], ll[idx][sel]] = True
    pred = P[K - 1]
    pq = post_q(D[p]["speed"], st["vcut"], T)
    rows.append(dict(split="train", panel=p, wid=gw, T=T, npred=int(pred.sum()), npost=int(pq.sum()),
                     hit=bool((pred & pq).any()), hit1=bool(near(pred, pq, 1).any()),
                     iou=iou(pred, y30), truth_hit=bool((y30 & pq).any()), ntrue=int(y30.sum()),
                     pred_links=np.flatnonzero(pred).tolist(), post_links=np.flatnonzero(pq).tolist(),
                     true_links=np.flatnonzero(y30).tolist()))
d = pd.DataFrame(rows)
d.to_pickle(str(WORK / "onset_post.pkl"))
tr = d[d.split == "train"]
print("train: truth-hit (true T+30 site queued at T+19..21)", round(tr.truth_hit.mean(), 3),
      "| pred hit", round(tr.hit.mean(), 3), "hit±1", round(tr.hit1.mean(), 3), "| IoU", round(tr.iou.mean(), 3),
      "| IoU when hit", round(tr[tr.hit].iou.mean(), 3), "when miss", round(tr[~tr.hit].iou.mean(), 3),
      "| post empty", round((tr.npost == 0).mean(), 3))
for sp_, x in d.groupby("split"):
    print(sp_, "n", len(x), "pred hit", round(x.hit.mean(), 3), "hit±1", round(x.hit1.mean(), 3), "post empty",
          round((x.npost == 0).mean(), 3))
    print("   per panel hit:", x.groupby("panel").hit.mean().round(2).to_dict())
